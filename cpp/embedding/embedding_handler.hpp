#pragma once

#include "../common/common.h"
#include "dense_mat.hpp"
#include "sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "partitioner.hpp"
#include "algo.hpp"
namespace hipgraph::distviz::embedding {

template<typename INDEX_TYPE, typename  VALUE_TYPE,size_t dimension>
class EmbeddingHandler {



public:

  Process3DGrid* grid;
  EmbeddingHandler<INDEX_TYPE,VALUE_TYPE,dimension>(Process3DGrid* grid){
    this->grid = grid;
  }




  void generate_embedding(vector<Tuple<VALUE_TYPE>>* input_graph,DenseMat<INDEX_TYPE, VALUE_TYPE, dimension>* dense_output,
                          uint64_t gRows, uint64_t gCols, uint64_t gNNZ, int batch_size,
                          int iterations, float lr, int nsamples, float alpha,float beta,
                          bool col_major=false, bool sync_comm=false, double drop_out_error_threshold=0,
                          int ns_generation_skip_factor=100, int repulsive_force_scaling_factor=2){

    auto localBRows = divide_and_round_up(gCols,grid->col_world_size);
    auto localARows = divide_and_round_up(gRows,grid->col_world_size);


    FileWriter<int,float,2> fileWriter;

    auto shared_sparseMat_receiver_coords = std::make_shared<std::vector<Tuple<VALUE_TYPE>>>(*input_graph);
    auto shared_sparseMat_sender_coords = std::make_shared<std::vector<Tuple<VALUE_TYPE>>>(*input_graph);


    auto shared_sparseMat = make_shared<SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,input_graph, gRows,gCols, gNNZ, batch_size,localARows, localBRows, false, false);

    auto shared_sparseMat_receiver = make_shared<SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,shared_sparseMat_receiver_coords.get(), gRows,gCols, gNNZ, batch_size,
                                                                                 localARows, localBRows, true, false);

    auto shared_sparseMat_sender = make_shared<SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,shared_sparseMat_sender_coords.get(), gRows,gCols, gNNZ, batch_size,
                                                                           localARows, localBRows, false, true);



    auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(grid));
    cout<<" rank  start partitioning data"<<grid->rank_in_col<<endl;

    partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(shared_sparseMat_sender.get());
    cout<<" rank  start shared_sparseMat_sender data"<<grid->rank_in_col<<endl;
    partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(shared_sparseMat_receiver.get());
    cout<<" rank  start shared_sparseMat_receiver data"<<grid->rank_in_col<<endl;
    partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(shared_sparseMat.get());



    cout<<" rank  after partitioning data"<<grid->rank_in_col<<" size "<<shared_sparseMat.get()->coords->size()<<endl;

    shared_sparseMat.get()->initialize_CSR_blocks();
    cout<<" rank "<<grid->rank_in_col<<" CSR shared_sparseMat initialization completed "<<shared_sparseMat.get()->coords->size()<<endl;

    CSRLocal<INDEX_TYPE, VALUE_TYPE> *csr_block = shared_sparseMat.get()->csr_local_data.get();
    CSRHandle<INDEX_TYPE, VALUE_TYPE> *csr_handle = csr_block->handler.get();

    std::vector<int>& row_offsets = csr_handle->rowStart;
    std::vector<int>& col_indices =  csr_handle->col_idx;
    std::vector<float>& values = csr_handle->values;


      cout<<" rank "<<grid->rank_in_col<<" CSR shared_sparseMat_receiver initialization started "<<shared_sparseMat.get()->coords->size()<<endl;
//      shared_sparseMat_receiver.get()->initialize_CSR_blocks();
      cout<<" rank "<<grid->rank_in_col<<" CSR shared_sparseMat_receiver initialization completed "<<shared_sparseMat.get()->coords->size()<<endl;

    shared_sparseMat_sender.get()->initialize_CSR_blocks();
    cout<<" rank "<<grid->rank_in_col<<" CSR  shared_sparseMat_sender initialization completed "<<shared_sparseMat.get()->coords->size()<<endl;



      CSRLocal<INDEX_TYPE, VALUE_TYPE> *csr_block_sender = shared_sparseMat_sender.get()->csr_local_data.get();
      CSRHandle<INDEX_TYPE, VALUE_TYPE> *csr_handle_sender = csr_block_sender->handler.get();

      std::vector<int>& row_offsets_send = csr_handle_sender->rowStart;
      std::vector<int>& col_indices_send =  csr_handle_sender->col_idx;
      std::vector<float>& values_send = csr_handle_sender->values;

    cout<<" rank "<<grid->rank_in_col<<" CSR shared_sparseMat_receiver initialization completed "<<shared_sparseMat.get()->coords->size()<<endl;
   // fileWriter.parallel_write_csr(grid,"/global/homes/i/isjarana/distviz_executions/perf_comparison/DistViz/MNIST/csr_native.txt",row_offsets,col_indices,values,shared_sparseMat_receiver.get()->proc_row_width);
   // fileWriter.parallel_write_csr(grid,"/global/homes/i/isjarana/distviz_executions/perf_comparison/DistViz/MNIST/csr_sender.txt",row_offsets_send,col_indices_send,values_send,shared_sparseMat_receiver.get()->proc_row_width,true);
    MPI_Barrier(grid->col_world);
   unique_ptr<EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>
        embedding_algo = unique_ptr<EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>(
                new EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>(
                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                    shared_sparseMat_sender.get(), dense_output, grid,
                    alpha, beta, 5, -5,col_major,sync_comm));
    embedding_algo.get()->algo_force2_vec_ns(iterations, batch_size, nsamples, lr,drop_out_error_threshold,ns_generation_skip_factor,repulsive_force_scaling_factor);

  }

  };
}

