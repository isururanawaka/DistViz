#pragma once

#include "lib/Mrpt_Sparse.h"
#include <iostream>
#include "Eigen/Dense"
#include "io/file_reader.hpp"
#include <chrono>
#include "knng/global_tree_handler.hpp"
#include "net/process_3D_grid.hpp"
#include "knng/knng_handler.hpp"
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>
#include <fstream>
#include <math.h>
#include <chrono>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <cstring>
#include "common/common.h"
#include "embedding/embedding_handler.hpp"
#include "embedding/dense_mat.hpp"

using namespace std;
using namespace std::chrono;
using namespace hipgraph::distviz::net;
using namespace hipgraph::distviz::knng;
using namespace hipgraph::distviz::embedding;
using namespace hipgraph::distviz::io;

int main(int argc, char *argv[]) {

    string input_path;
    string output_path;
    long data_set_size = 0;
    int dimension = 0;
    int ntrees = 10;
    int tree_depth = 0;
    double density = 0;
    int nn = 0;
    double tree_depth_ratio = 1;
    bool use_locality_optimization = false;
    int local_tree_offset = 2;
    int file_format = 0;
    int data_starting_index = 8;
    bool generate_knng_output = false;
    float target_local_recall = 0.99;

    float alpha = 0;
    float beta = 1;
    bool col_major = false;
    bool sync_comm = true;
    int iterations = 1200;
    int batch_size = 256;
    bool full_batch_training = true;

    bool sparse_input = false;
    float lr = 0.2;
    int nsamples = 5;

    double drop_out_error_threshold = 0;

    const int embedding_dimension = 2;

    int bytes_for_data_type = 4;

    int file_offset = 8;

    bool skip_auto_tune = false;

    int repulsive_force_scaling_factor = 2;
    int ns_generation_skip_factor = 100;

    bool skip_knng = false;

    for (int p = 0; p < argc; p++) {
        if (strcmp(argv[p], "-input") == 0) {
            input_path = argv[p + 1];
        } else if (strcmp(argv[p], "-output") == 0) {
            output_path = argv[p + 1];
        } else if (strcmp(argv[p], "-data-set-size") == 0) {
            data_set_size = atol(argv[p + 1]);
        } else if (strcmp(argv[p], "-dimension") == 0) {
            dimension = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-ntrees") == 0) {
            ntrees = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-tree-depth-ratio") == 0) {
            tree_depth_ratio = stof(argv[p + 1]);
        } else if (strcmp(argv[p], "-density") == 0) {
            density = strtod(argv[p+1], NULL);
        } else if (strcmp(argv[p], "-nn") == 0) {
            nn = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-locality") == 0) {
            use_locality_optimization = atoi(argv[p + 1]) == 1 ? true : false;
        } else if (strcmp(argv[p], "-local-tree-offset") == 0) {
            local_tree_offset = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-data-file-format") == 0) {
            file_format = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-bytes-for-data-type") == 0) {
            bytes_for_data_type = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-data-starting-index") == 0) {
            data_starting_index = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-generate-knng-output") == 0) {
            int generate = atoi(argv[p + 1]);
            generate_knng_output = generate == 1 ? true : false;
        } else if (strcmp(argv[p], "-target-local-recall") == 0) {
            target_local_recall = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-alpha") == 0) {
            alpha = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-beta") == 0) {
            beta = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-col-major") == 0) {
            int col_major_input = atoi(argv[p + 1]);
            col_major = col_major_input == 1 ? true : false;
        } else if (strcmp(argv[p], "-sync-comm:") == 0) {
            int sync_comm_input = atoi(argv[p + 1]);
            sync_comm = sync_comm_input == 1 ? true : false;
        } else if (strcmp(argv[p], "-nsamples") == 0) {
            nsamples = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-lr") == 0) {
            lr = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-iterations") == 0) {
            iterations = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-full_batch_training") == 0) {
            int full_batch_tra = atoi(argv[p + 1]);
            full_batch_training = full_batch_tra == 1 ? true : false;
        } else if (strcmp(argv[p], "-batch") == 0) {
            batch_size = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-dropout-error-th") == 0) {
            drop_out_error_threshold = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-file-offset") == 0) {
            file_offset = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-sparse-input") == 0) {
            sparse_input = atoi(argv[p + 1]) == 1 ? true : false;
        } else if (strcmp(argv[p], "-skip-auto-tune") == 0) {
            skip_auto_tune = atoi(argv[p + 1]) == 1 ? true : false;
        } else if (strcmp(argv[p], "-ns_generation_skip_factor") == 0) {
            ns_generation_skip_factor = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-repulsive_force_scaling_factor") == 0) {
            repulsive_force_scaling_factor = atof(argv[p + 1]);
        }
    }

    if (input_path.size() == 0) {
        printf("Valid input path needed!...\n");
        exit(1);
    }

    if (output_path.size() == 0) {
        printf("Valid out path needed!...\n");
        exit(1);
    }

    if (data_set_size == 0) {
        printf("Dataset size should be greater than 0\n");
        exit(1);
    }

    if (dimension == 0) {
        printf("Dimension size should be greater than 0\n");
        exit(1);
    }

    if (nn == 0) {
        printf("Nearest neighbours size should be greater than 0\n");
        exit(1);
    }

    if (density == 0) {
        density = 1.0 / sqrt(dimension);
    }

    cout<<" density "<<density<<endl;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (tree_depth == 0) {
        tree_depth = log2(size) + 2;
    }

    //Initialize custom MPI datatypes
    initialize_mpi_datatypes<int, float, embedding_dimension>();

    //process grid used in the distributed setup
    auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(size, 1, 1, 1));

    //original datamatrix for dense data
    unique_ptr <ValueType2DVector<float>> data_matrix_ptr = make_unique<ValueType2DVector<float>>();

    //original datamatrix for sparse data
    Eigen::SparseMatrix<float, Eigen::RowMajor> sparse_matrix;


    FileWriter<int, float, embedding_dimension> fileWriter;

    auto t = start_clock();
    if (file_format == 0) {
        FileReader<int, float, 2>::ubyte_read(
                input_path, data_matrix_ptr.get(), data_set_size, dimension,
                grid.get()->rank_in_col, grid.get()->col_world_size);
    } else if (file_format == 1) {
        FileReader<int, float, 2>::fvecs_read(
                input_path, data_matrix_ptr.get(), data_set_size, dimension,
                grid.get()->rank_in_col, grid.get()->col_world_size);
    } else if (file_format == 2) {
        if (sparse_input) {
            sparse_matrix =
                    Eigen::SparseMatrix<float, Eigen::RowMajor>(dimension, data_set_size);
            FileReader<uint64_t, float, 2>::read_fbin_sparse(
                    input_path, sparse_matrix, data_set_size, dimension,
                    grid.get()->rank_in_col, grid.get()->col_world_size, file_offset);
        } else {
            FileReader<uint64_t, float, 2>::read_fbin(
                    input_path, data_matrix_ptr.get(), data_set_size, dimension,
                    grid.get()->rank_in_col, grid.get()->col_world_size, file_offset);
//            fileWriter.parallel_write_2D("/pscratch/sd/i/isjarana/benchmarking/inputs/laborflow/1024/data.txt",data_matrix_ptr.get());
//            return 0;

        }
    } else if (file_format == 3) {
        skip_knng = true;
    } else if (file_format == 4) {
        FileReader<int, float, 2>::read_txt(
                input_path, data_matrix_ptr.get(), data_set_size, dimension,
                grid.get()->rank_in_col, grid.get()->col_world_size);
    } else if (file_format == 5) {
        FileReader<int, float, 2>::read_ubin(
                input_path, data_matrix_ptr.get(), data_set_size, dimension,
                grid.get()->rank_in_col, grid.get()->col_world_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    stop_clock_and_add(t, "IO Time");
    t = start_clock();
    std::cout << "calling data loading completed " << rank << " " << std::endl;
    std::cout << "calling grow trees" << rank << " " << std::endl;

    shared_ptr < vector < Tuple<float>>> knng_graph_ptr = make_shared < vector < Tuple<float>>>();
    Eigen::MatrixXf data_matrix = Eigen::MatrixXf();

    if (skip_knng) {
        FileReader<int, float, 2>::parallel_read_MM(input_path, knng_graph_ptr.get(), false);
    } else {
        std::cout << "calling KNNGHandler rank " << rank << " with input matrix:  "
                  << (sparse_input ? sparse_matrix.rows() : data_matrix_ptr->size())
                  << "*"
                  << (sparse_input ? sparse_matrix.cols()
                                   : (*data_matrix_ptr)[0].size())
                  << std::endl;

        auto data_set_size_local = sparse_input ? sparse_matrix.rows() : data_matrix_ptr->size();
        auto knng_handler = unique_ptr<KNNGHandler<int, float>>(
                new KNNGHandler<int, float>(ntrees, tree_depth, tree_depth_ratio,
                                            local_tree_offset, data_set_size,
                                            data_set_size_local, dimension,
                                            grid.get()));

        if (grid.get()->col_world_size == 1) {
            if (sparse_input) {
                knng_handler.get()->build_local_KNNG_Sparse(
                        sparse_matrix, knng_graph_ptr.get(), nn, target_local_recall,
                        generate_knng_output, output_path + "/knng.txt", true, density,
                        skip_auto_tune);
            } else {
                data_matrix = Eigen::MatrixXf((*data_matrix_ptr)[0].size(),
                                              (*data_matrix_ptr).size());
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (*data_matrix_ptr).size(); ++i) {
                    for (int j = 0; j < (*data_matrix_ptr)[0].size(); ++j) {
                        data_matrix(j, i) = (*data_matrix_ptr)[i][j];
                    }
                }
                knng_handler.get()->build_local_KNNG(
                        data_matrix, knng_graph_ptr.get(), nn, target_local_recall,
                        generate_knng_output, output_path + "/knng.txt", true, density);
            }
        } else {
            knng_handler.get()->build_distributed_KNNG(
                    data_matrix_ptr.get(), knng_graph_ptr.get(), density,
                    use_locality_optimization, nn, target_local_recall,
                    generate_knng_output, output_path + "/knng.txt");
        }
    }

    std::cout << "calling grow trees completed" << rank << " " << std::endl;
    stop_clock_and_add(t, "KNNG Total Time");

    t = start_clock();
    cout << " rank " << rank << " output size: " << knng_graph_ptr.get()->size() << endl;

    auto localARows = divide_and_round_up(data_set_size, grid.get()->col_world_size);

    if (full_batch_training)
        batch_size = localARows;

    auto dense_mat = shared_ptr<DenseMat<int, float, embedding_dimension>>(
            new DenseMat<int, float, embedding_dimension>(grid.get(), localARows));

//  dense_mat->print_matrix_rowptr(0);

    auto embedding_handler = unique_ptr<EmbeddingHandler<int, float, embedding_dimension>>(
            new EmbeddingHandler<int, float, embedding_dimension>(grid.get()));

    auto gNNZ = data_set_size * nn;
    std::cout << "start generating embedding " << rank << " rows " << localARows << " gNNZ " << gNNZ << std::endl;
     MPI_Barrier(grid->col_world);
    embedding_handler->generate_embedding(knng_graph_ptr.get(), dense_mat.get(),
                                          data_set_size, data_set_size, gNNZ,
                                          batch_size, iterations, lr, nsamples, alpha, beta,
                                          col_major, sync_comm, drop_out_error_threshold, ns_generation_skip_factor,
                                          repulsive_force_scaling_factor);

    std::cout << "stop generating embedding " << rank << " " << std::endl;
    stop_clock_and_add(t, "Embedding Total Time");

    t = start_clock();

    fileWriter.parallel_write(output_path + "/embedding.txt", dense_mat.get()->nCoordinates, localARows,
                              embedding_dimension);
    stop_clock_and_add(t, "IO Time");

    ofstream fout;
    fout.open("perf_output", std::ios_base::app);

    json j_obj;
    j_obj["algo"] = "DistViz";
    j_obj["p"] = grid->col_world_size;
    j_obj["k"] = nn;
    j_obj["perf_stats"] = json_perf_statistics();
    if (rank == 0) {
        fout << j_obj.dump(4) << "," << endl;
    }
    fout.close();

    MPI_Finalize();
}
