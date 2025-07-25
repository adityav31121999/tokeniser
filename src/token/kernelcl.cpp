#ifdef USE_OPENCL
#include "include/tokenise.hpp"
#include <chrono>

// host side function for embeddings
void tokeniser::clEmbeddingFormula(OpenCLContext& ocl_context, std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds_ignored, int& d_dim, 
    int& vocSize_val, float r1) 
{
    if (!ocl_context.context() || !ocl_context.queue()) { // Use () for cl.hpp accessors
        std::cerr << "OpenCL context or command queue not initialized via singleton." << std::endl;
        return;
    }

    // Resize embedding vector to hold the results
    embedding.resize(vocSize_val, std::vector<float>(d_dim));
    // Calculate total number of elements
    size_t total_elements = (size_t)vocSize_val * d_dim;
    if (total_elements == 0) return;
    cl_int err;

    // Create a flat array for host memory, then copy back to 2D vector
    std::vector<float> flat_embeddings(total_elements);
    // Create device buffer
    cl::Buffer embeddings_buffer(ocl_context.context, CL_MEM_WRITE_ONLY, sizeof(float) * total_elements, NULL, &err);
    CHECK_CL(err);
    // Create kernel object
    cl::Kernel kernel(ocl_context.embeddingProgram, "generate_embeddings", &err);
    CHECK_CL(err);

    // Set Kernel Arguments
    kernel.setArg(0, embeddings_buffer);
    kernel.setArg(1, d_dim);
    kernel.setArg(2, r1);
    unsigned int initial_seed_offset = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    kernel.setArg(4, initial_seed_offset);

    // Execute Kernel
    cl::NDRange global_work_size(total_elements);
    cl::NDRange local_work_size = cl::NullRange; // Let OpenCL decide optimal local size

    err = ocl_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);
    CHECK_CL(err);
    // Read Results Back
    err = ocl_context.queue.enqueueReadBuffer(embeddings_buffer, CL_TRUE, 0,
                                              sizeof(float) * total_elements, flat_embeddings.data());
    CHECK_CL(err);

    // Copy flat_embeddings to the 2D embedding vector
    for (int i = 0; i < vocSize_val; ++i) {
        for (int j = 0; j < d_dim; ++j) {
            embedding[i][j] = flat_embeddings[i * d_dim + j];
        }
    }
    // No explicit release needed for cl.hpp objects as they manage resources via RAII
}

// The clVectorInverse function would follow a similar pattern, using inverseProgram
void tokeniser::clVectorInverse(OpenCLContext& ocl_context, std::vector<std::vector<float>>& deEmbedding, \
    const std::vector<std::vector<float>>& embedding, int& d_dim, int& vocSize_val) 
{
    if (!ocl_context.context() || !ocl_context.queue()) {
        std::cerr << "OpenCL context or command queue not initialized via singleton." << std::endl;
        return;
    }

    deEmbedding.resize(vocSize_val, std::vector<float>(d_dim));
    size_t total_elements = (size_t)vocSize_val * d_dim;
    if (total_elements == 0) return;

    cl_int err;

    // Flatten input embedding data for transfer
    std::vector<float> flat_embedding_input(total_elements);
    for (int i = 0; i < vocSize_val; ++i) {
        for (int j = 0; j < d_dim; ++j) {
            flat_embedding_input[i * d_dim + j] = embedding[i][j];
        }
    }

    cl::Buffer input_buffer(ocl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * total_elements, flat_embedding_input.data(), &err);
    CHECK_CL(err);

    cl::Buffer output_buffer(ocl_context.context, CL_MEM_WRITE_ONLY,
                             sizeof(float) * total_elements, NULL, &err);
    CHECK_CL(err);

    cl::Kernel kernel(ocl_context.inverseProgram, "batchedVectorInverseKernel", &err);
    CHECK_CL(err);

    // Determine local work size for reduction kernel
    // The kernel is designed with a single work-group per row (vector) in mind.
    // The local work size should be at least 'd_dim' or a power of 2 greater than 'd_dim'
    // for the reduction, up to CL_DEVICE_MAX_WORK_GROUP_SIZE.
    // For simplicity, let's pick a common value or max_work_group_size if d_dim is small.
    // You might want to query device capabilities for optimal local_work_size.
    size_t local_work_size_x = 256; // Example. Adjust for your hardware/kernel design
    if (d_dim < local_work_size_x) {
        // Round up to the next power of 2 if not already one
        local_work_size_x = 1;
        while (local_work_size_x < d_dim) local_work_size_x <<= 1;
        if (local_work_size_x > 256) local_work_size_x = 256; // Cap to common max
    }


    kernel.setArg(0, output_buffer);
    kernel.setArg(1, input_buffer);
    kernel.setArg(2, vocSize_val); // N is number of rows
    kernel.setArg(3, d_dim);       // d is dimension of each vector
    kernel.setArg(4, sizeof(float) * local_work_size_x, NULL); // Local memory for s_data

    // Global work size: Each row/vector needs a work-group operating on 'd_dim' elements
    // The kernel uses get_group_id(1) for rows, so global_work_size should map to that.
    // Global X: local_work_size_x * number of groups in X (which is 1 per row for the col dim)
    // Global Y: vocSize_val (number of rows)
    cl::NDRange global_size_2d(local_work_size_x, vocSize_val); // Corrected global size for 2D kernel
    cl::NDRange local_size_2d(local_work_size_x, 1); // Local size: X for columns, 1 for rows (each row a group)

    err = ocl_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size_2d, local_size_2d);
    CHECK_CL(err);

    std::vector<float> flat_deEmbedding_output(total_elements);
    err = ocl_context.queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0,
                                              sizeof(float) * total_elements, flat_deEmbedding_output.data());
    CHECK_CL(err);

    // Copy flat_deEmbedding_output to the 2D deEmbedding vector
    for (int i = 0; i < vocSize_val; ++i) {
        for (int j = 0; j < d_dim; ++j) {
            deEmbedding[i][j] = flat_deEmbedding_output[i * d_dim + j];
        }
    }
}


#endif // USE_OPENCL
