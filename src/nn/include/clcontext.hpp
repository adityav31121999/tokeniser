#ifdef USE_OPENCL
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl.hpp> 
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

// Helper to check for OpenCL errors
inline void check_cl(cl_int err, const char* file, int line) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error: %d at %s:%d\n", err, file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CL(err) check_cl(err, __FILE__, __LINE__)

// source for token embedding
const std::string embeddingFormulaSource = R"CLC(
    // Basic XorShift32 PRNG for OpenCL
    // seed must be unique per work-item
    unsigned int xorshift32(unsigned int x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    // Function to convert a normalized [0,1] float to a custom range [r1, r2]
    float scale_random(unsigned int* seed_ptr, float r1, float r2) {
        // Generate a random unsigned int
        *seed_ptr = xorshift32(*seed_ptr);
        // Convert to float in [0, 1] range
        float normalized_val = (float)(*seed_ptr) / (float)0xFFFFFFFFU;
        // Scale to [r1, r2]
        return r1 + normalized_val * (r2 - r1);
    }

    __kernel void generate_embeddings(
        __global float* embeddings_out,
        const int d_dim,
        const float r1,
        const float r2,
        const unsigned int initial_seed_offset) {

        int global_id = get_global_id(0); // 0-indexed global work-item ID
        // int total_elements = get_global_size(0); // Not directly used in this snippet

        // --- MISSING LINE ADDED HERE ---
        unsigned int seed = initial_seed_offset + global_id + 1; // Declare and initialize 'seed' for this work-item

        // Generate random float in [r1, r2] using the local 'seed'
        embeddings_out[global_id] = scale_random(&seed, r1, r2);
    }
)CLC";


const std::string vectorInverseSource = R"CLC(
    __kernel void batchedVectorInverseKernel(
        __global float* output, __global const float* input, 
        const int N, const int d, __local float* s_data)
    {
        // Identify which row (vector) this work-group is working on.
        const int row_idx = get_group_id(1);

        // Identify the thread's local and global column indices.
        const int tid_in_block = get_local_id(0);
        const int col_idx = get_global_id(0);

        // --- Step 1: Parallel Reduction to find the squared magnitude ---
        float my_val = 0.0f;
        if (col_idx < d) {
            my_val = input[row_idx * d + col_idx];
        }
        
        s_data[tid_in_block] = my_val * my_val;

        // Synchronize to make sure all work-items have written to local memory.
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the reduction in local memory.
        for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
            if (tid_in_block < s) {
                s_data[tid_in_block] += s_data[tid_in_block + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // The first work-item holds the final result.
        const float squared_magnitude = s_data[0];

        // --- Step 2: Element-wise division ---
        if (col_idx < d) {
            if (squared_magnitude > 1e-9f) {
                output[row_idx * d + col_idx] = my_val / squared_magnitude;
            } else {
                output[row_idx * d + col_idx] = 0.0f;
            }
        }
    }
)CLC";


// Singleton to manage the global OpenCL context, device, and queue
class OpenCLContext {
public:
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    cl::Program embeddingProgram;
    cl::Program inverseProgram;

    static OpenCLContext& getInstance() {
        static OpenCLContext instance;
        return instance;
    }

    OpenCLContext() {
        // Find platforms and devices
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if (all_platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }
        cl::Platform default_platform = all_platforms[0];

        std::vector<cl::Device> all_devices;
        default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
        if (all_devices.empty()) {
            throw std::runtime_error("No OpenCL GPU devices found.");
        }
        device = all_devices[0];

        // Create context and queue
        context = cl::Context({device});
        queue = cl::CommandQueue(context, device);

        // Compile the kernels
        embeddingProgram = cl::Program(context, embeddingFormulaSource);
        inverseProgram = cl::Program(context, vectorInverseSource);

        cl_int err;
        err = embeddingProgram.build({device});
        if (err != CL_SUCCESS) {
            std::string log = embeddingProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "OpenCL Kernel Build Error (embeddingProgram):\n" << log << std::endl;
            exit(EXIT_FAILURE);
        }

        err = inverseProgram.build({device});
        if (err != CL_SUCCESS) {
            std::string log = inverseProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "OpenCL Kernel Build Error (inverseProgram):\n" << log << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Since this is a singleton, delete copy/move to prevent unintended copies
    OpenCLContext(const OpenCLContext&) = default;
    OpenCLContext& operator=(const OpenCLContext&) = default;
    OpenCLContext(OpenCLContext&&) = default;
    OpenCLContext& operator=(OpenCLContext&&) = default;
};

#endif