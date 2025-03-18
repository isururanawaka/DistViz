#pragma once
#include "../common//common.h"
#include "../embedding/dense_mat.hpp"
#include "../embedding/sparse_mat.hpp"
#include "process_3D_grid.hpp"
#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <vector>
#include <thread>
#include <chrono>
#include "../embedding/csr_local.hpp"

using namespace hipgraph::distviz::common;
using namespace hipgraph::distviz::embedding;

namespace hipgraph::distviz::net {

/**
 * This class represents the data transfer related operations across processes
 * based on internal data connectivity patterns. This implementation is tightly coupled
 * with 1D partitioning of Sparse Matrix and 1D partitioning of Dense Matrix.
 */

template <typename SPT, typename DENT, size_t embedding_dim> class DataComm {

private:
  hipgraph::distviz::embedding::SpMat<SPT,DENT> *sp_local_receiver;
  hipgraph::distviz::embedding::SpMat<SPT,DENT> *sp_local_sender;
  DenseMat<SPT, DENT, embedding_dim> *dense_local;
  Process3DGrid *grid;
  shared_ptr<vector<int>> sdispls;
  shared_ptr<vector<int>> sendcounts;
  shared_ptr<vector<int>> rdispls;
  shared_ptr<vector<int>> receivecounts;
  shared_ptr<vector<unordered_set<uint64_t>>> receive_col_ids_list;
  shared_ptr<vector<unordered_set<uint64_t>>> send_col_ids_list;
  shared_ptr<unordered_map<uint64_t, unordered_map<int,bool>>> send_indices_to_proc_map;
  shared_ptr<unordered_map<uint64_t, unordered_map<int,bool>>> receive_indices_to_proc_map;
  unique_ptr<vector<SPT>> sendbuf_ids = unique_ptr<vector<SPT>>(new vector<SPT>());
  unique_ptr<vector<SPT>> receivebuf_ids = unique_ptr<vector<SPT>>(new vector<SPT>());

  int batch_id;

  double alpha;

public:

  shared_ptr<vector<int>> receive_counts_cyclic;
  shared_ptr<vector<int>> rdispls_cyclic;
  shared_ptr<vector<int>> send_counts_cyclic;
  shared_ptr<vector<int>> sdispls_cyclic;

  MPI_Request request = MPI_REQUEST_NULL;


  DataComm(hipgraph::distviz::embedding::SpMat<SPT,DENT> *sp_local_receiver,
           hipgraph::distviz::embedding::SpMat<SPT,DENT> *sp_local_sender,
           DenseMat<SPT, DENT, embedding_dim> *dense_local, Process3DGrid *grid,
           int batch_id, double alpha) {
    this->sp_local_receiver = sp_local_receiver;
    this->sp_local_sender = sp_local_sender;
    this->dense_local = dense_local;
    this->grid = grid;
    this->sdispls = make_shared<vector<int>>(grid->world_size, 0);
    this->sendcounts = make_shared<vector<int>>(grid->world_size, 0);
    this->rdispls = make_shared<vector<int>>(grid->world_size, 0);
    this->receivecounts = make_shared<vector<int>>(grid->world_size, 0);
    this->send_counts_cyclic = make_shared<vector<int>>(grid->world_size, 0);
    this->receive_counts_cyclic = make_shared<vector<int>>(grid->world_size, 0);
    this->sdispls_cyclic = make_shared<vector<int>>(grid->world_size, 0);
    this->rdispls_cyclic = make_shared<vector<int>>(grid->world_size, 0);
    this->receive_col_ids_list = make_shared<vector<unordered_set<uint64_t>>>(grid->world_size);
    this->send_col_ids_list = make_shared<vector<unordered_set<uint64_t>>>(grid->world_size);
    this->send_indices_to_proc_map = make_shared<unordered_map<uint64_t, unordered_map<int,bool>>>();
    this->receive_indices_to_proc_map = make_shared<unordered_map<uint64_t, unordered_map<int,bool>>>();
    this->batch_id = batch_id;
    this->alpha = alpha;



  }

  ~DataComm() {}

  void onboard_data(hipgraph::distviz::embedding::SpMat<SPT,DENT> communication_csr=nullptr) {

    int total_send_count = 0;
    // processing chunks
    // calculating receiving data cols
    if (communication_csr){





}else{

    if (alpha==0) {
      // This represents the case for pulling
      cout<<" rank "<<grid->rank_in_col<<" filling col values "<<endl;
         this->sp_local_receiver->fill_col_ids(batch_id, 0, grid->col_world_size, receive_col_ids_list.get(),receive_indices_to_proc_map.get(), 0);
         cout<<" rank "<<grid->rank_in_col<<" receiver values completed "<<endl;
         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id,0,grid->col_world_size, send_col_ids_list.get(),send_indices_to_proc_map.get(), 0);
         cout<<" rank "<<grid->rank_in_col<<" sender values completed "<<endl;
    } else if (alpha == 1.0) {
      // This represents the case for pushing
         this->sp_local_receiver->fill_col_ids(batch_id, 0, grid->col_world_size, receive_col_ids_list.get(),receive_indices_to_proc_map.get(), 1);

         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id,0,grid->col_world_size, send_col_ids_list.get(),send_indices_to_proc_map.get(), 1);
    } else if (alpha> 0 and alpha < 1.0){

      // This represents the case for pull and pushing
          int end_process = get_end_proc(1,alpha, grid->col_world_size);

         this->sp_local_receiver->fill_col_ids(batch_id, 1, end_process, receive_col_ids_list.get(),receive_indices_to_proc_map.get(), 1);

         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id,1,end_process, send_col_ids_list.get(),send_indices_to_proc_map.get(), 1);

      if (batch_id>=0) {
         this->sp_local_receiver->fill_col_ids(batch_id, end_process, grid->col_world_size, receive_col_ids_list.get(),receive_indices_to_proc_map.get(), 0);

         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id, end_process,grid->col_world_size, send_col_ids_list.get(),send_indices_to_proc_map.get(), 0);
      }

    } else {
      cout<<"  alpha needs to be in the range of [0,1]"<<endl;
    }

    for (int i = 0; i < grid->world_size; i++) {
        (*receivecounts)[i] = (*receive_col_ids_list)[i].size();
        (*sendcounts)[i] = (*send_col_ids_list)[i].size();
    }
}
  }

 inline void transfer_data(std::vector<DataTuple<DENT, embedding_dim>> *sendbuf_cyclic,
                            std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                            bool synchronous,MPI_Request* req, int iteration,
                            int batch_id, int starting_proc, int end_proc,
                            bool temp_cache) {

    int total_receive_count = 0;
    vector<int> offset_vector(grid->col_world_size, 0);

    int total_send_count = 0;
    send_counts_cyclic = make_shared<vector<int>>(grid->col_world_size, 0);
    receive_counts_cyclic = make_shared<vector<int>>(grid->col_world_size, 0);
    sdispls_cyclic = make_shared<vector<int>>(grid->col_world_size, 0);
    rdispls_cyclic = make_shared<vector<int>>(grid->col_world_size, 0);

    vector<int> sending_procs;
    vector<int> receiving_procs;

    for (int i = starting_proc; i < end_proc; i++) {
      int sending_rank = (grid->rank_in_col + i) % grid->col_world_size;
      int receiving_rank =
          (grid->rank_in_col >= i)
              ? (grid->rank_in_col - i) % grid->col_world_size
              : (grid->col_world_size - i + grid->rank_in_col) % grid->col_world_size;
      sending_procs.push_back(sending_rank);
      receiving_procs.push_back(receiving_rank);

    }

    for (int i = 0; i < sending_procs.size(); i++) {
      (*send_counts_cyclic)[sending_procs[i]] = (*sendcounts)[sending_procs[i]];
      (*receive_counts_cyclic)[receiving_procs[i]] =
          (*receivecounts)[receiving_procs[i]];
      total_send_count += (*send_counts_cyclic)[sending_procs[i]];
      total_receive_count += (*receive_counts_cyclic)[receiving_procs[i]];
    }

    for (int i = 0; i < grid->col_world_size; i++) {
      (*sdispls_cyclic)[i] =
          (i > 0) ? (*sdispls_cyclic)[i - 1] + (*send_counts_cyclic)[i - 1]
                  : (*sdispls_cyclic)[i];
      (*rdispls_cyclic)[i] =
          (i > 0) ? (*rdispls_cyclic)[i - 1] + (*receive_counts_cyclic)[i - 1]
                  : (*rdispls_cyclic)[i];
    }


    if (total_send_count > 0) {
      sendbuf_cyclic->resize(total_send_count);
      for (const auto &pair : (*send_indices_to_proc_map)) {
        auto col_id = pair.first;
        bool already_fetched = false;
        std::array<DENT, embedding_dim> dense_vector;
        for (int i = 0; i < sending_procs.size(); i++) {
          if (pair.second.count(sending_procs[i]) > 0) {
            if (!already_fetched) {
              dense_vector = (this->dense_local)->fetch_local_data(col_id);
              already_fetched = true;
            }
            int offset = (*sdispls_cyclic)[sending_procs[i]];
            int index = offset_vector[sending_procs[i]] + offset;
            (*sendbuf_cyclic)[index].col =
                col_id + (this->sp_local_sender->proc_col_width *
                          this->grid->global_rank);
            (*sendbuf_cyclic)[index].value = dense_vector;
            offset_vector[sending_procs[i]]++;
          }
        }
      }
    }

    if (total_receive_count>0) {
      receivebuf->resize(total_receive_count);
    }

    if (synchronous) {
      auto t = start_clock();
      MPI_Alltoallv((*sendbuf_cyclic).data(), (*send_counts_cyclic).data(),
                    (*sdispls_cyclic).data(), DENSETUPLE, (*receivebuf).data(),
                    (*receive_counts_cyclic).data(), (*rdispls_cyclic).data(),
                    DENSETUPLE, grid->col_world);
      MPI_Request dumy;
      this->populate_cache(sendbuf_cyclic, receivebuf, &dumy, true, iteration, batch_id,temp_cache);
      stop_clock_and_add(t, "Embedding Communication Time");
    }
  }

  void transfer_data(vector<uint64_t> col_ids, int iteration, int batch_id) {

    vector<vector<uint64_t>> receive_col_ids_list(grid->col_world_size);
    vector<uint64_t> send_col_ids_list;

    int total_send_count = 0;
    int total_receive_count = 0;

    for (int i = 0; i < col_ids.size(); i++) {
      int owner_rank = col_ids[i] / (this->sp_local_receiver)->proc_row_width;
      if (owner_rank == grid->rank_in_col) {
        send_col_ids_list.push_back(col_ids[i]);
      } else {
        receive_col_ids_list[owner_rank].push_back(col_ids[i]);
      }
    }

    for (int i = 0; i < grid->col_world_size; i++) {
       total_send_count = send_col_ids_list.size();
      if (i != grid->rank_in_col) {
        (*sendcounts)[i] = total_send_count;
      } else {
        (*sendcounts)[i] = 0;
      }
      (*receive_counts_cyclic)[i] = receive_col_ids_list[i].size();
    }

    (*sdispls)[0] = 0;
    (*rdispls_cyclic)[0] = 0;
    for (int i = 0; i < grid->col_world_size; i++) {

      (*sdispls)[i] = 0;
      (*rdispls_cyclic)[i] =
          (i > 0) ? (*rdispls_cyclic)[i - 1] + (*receive_counts_cyclic)[i - 1]
                  : (*rdispls_cyclic)[i];
      total_receive_count = total_receive_count + (*receive_counts_cyclic)[i];
    }

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> sendbuf =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    sendbuf->resize(total_send_count);

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> receivebuf_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    receivebuf_ptr.get()->resize(total_receive_count);


    for (int j = 0; j < send_col_ids_list.size(); j++) {
      int local_key =
          send_col_ids_list[j] -
          (grid->rank_in_col) * (this->sp_local_receiver)->proc_row_width;
      std::array<DENT, embedding_dim> val_arr =
          (this->dense_local)->fetch_local_data(local_key);
        (*sendbuf)[j].col = send_col_ids_list[j];
        (*sendbuf)[j].value = val_arr;
    }

    auto t = start_clock();
    MPI_Alltoallv((*sendbuf).data(), (*sendcounts).data(), (*sdispls).data(),
                  DENSETUPLE, (*receivebuf_ptr.get()).data(),
                  (*receive_counts_cyclic).data(), (*rdispls_cyclic).data(),
                  DENSETUPLE, grid->col_world);
    stop_clock_and_add(t, "Embedding Communication Time");
    MPI_Request dumy;
    this->populate_cache(sendbuf.get(),receivebuf_ptr.get(), &dumy, true, iteration, batch_id,true); // we should not do this
  }


void transfer_data(CSRLocal<SPT, DENT> *csr_block,  int iteration, int batch_id, bool shift=true) {
    CSRHandle<SPT, DENT>* data_handler = csr_block->handler.get();
    int total_send_count = 0;
    int total_receive_count = 0;
    if((iteration==0) or shift){
    sendcounts = make_shared<vector<int>> (grid->col_world_size, 0);
    sdispls = make_shared<vector<int>> (grid->col_world_size, 0);
    //unique_ptr<vector<int>> receive_counts_cyclic = unique_ptr<vector<int>>(new vector<int>(grid->col_world_size, 0));
   // unique_ptr<vector<int>> rdispls_cyclic = unique_ptr<vector<int>>(new vector<int>(grid->col_world_size, 0));
    receive_counts_cyclic = make_shared<vector<int>>(grid->col_world_size, 0);
    rdispls_cyclic = make_shared<vector<int>>(grid->col_world_size, 0);

    #pragma omp parallel for
    for (int proc = 0; proc < grid->col_world_size; proc++) {
        int start_index = proc * this->sp_local_receiver->proc_row_width;
        int end_index = (proc + 1) * this->sp_local_receiver->proc_row_width;
        int gRows = static_cast<int>(this->sp_local_receiver->gRows);
        if (proc == grid->col_world_size - 1) {
            end_index = min(end_index, gRows);
        }
        for (int i = start_index; i < end_index; i++) {
            if ((data_handler->rowStart[i + 1] - data_handler->rowStart[i]) > 0) {
                int id = shift?(i+iteration ) % this->sp_local_receiver->gRows:(i ) % this->sp_local_receiver->gRows;
                int target_proc = id/ this->sp_local_receiver->proc_row_width;
                if (target_proc != grid->rank_in_col) {
                    #pragma omp atomic
                    (*sendcounts)[target_proc]++;
                }
            }
        }
    }

    for (int proc = 0; proc < grid->col_world_size; proc++) {
        total_send_count += (*sendcounts)[proc];
    }

  //  unique_ptr<vector<SPT>> sendbuf_ids = unique_ptr<vector<SPT>>(new vector<SPT>());
    sendbuf_ids->resize(total_send_count);

  //  unique_ptr<vector<SPT>> receivebuf_ids = unique_ptr<vector<SPT>>(new vector<SPT>());

    MPI_Alltoall((*sendcounts).data(), 1, MPI_INT, receive_counts_cyclic->data(), 1, MPI_INT, grid->col_world);
    for (int i = 0; i < grid->col_world_size; i++) {
        (*sdispls)[i] = (i > 0) ? (*sdispls)[i - 1] + (*sendcounts)[i - 1] : 0;
        (*rdispls_cyclic)[i] = (i > 0) ? (*rdispls_cyclic)[i - 1] + (*receive_counts_cyclic)[i - 1] : 0;
        total_receive_count += (*receive_counts_cyclic)[i];
    }

    receivebuf_ids->resize(total_receive_count);

    vector<int> current_offset(grid->col_world_size, 0);

     #pragma omp parallel for
    for (int proc = 0; proc < grid->col_world_size; proc++) {
        int start_index = proc * this->sp_local_receiver->proc_row_width;
        int end_index = (proc + 1) * this->sp_local_receiver->proc_row_width;
        int gRows = static_cast<int>(this->sp_local_receiver->gRows);
        if (proc == grid->col_world_size - 1) {
            end_index = min(end_index, gRows);
        }
        for (int i = start_index; i < end_index; i++) {
            if ((data_handler->rowStart[i + 1] - data_handler->rowStart[i]) > 0) {
                int id = shift?(i+iteration ) % this->sp_local_receiver->gRows: i% this->sp_local_receiver->gRows;
                int target_proc = id / this->sp_local_receiver->proc_row_width;
                if (target_proc != grid->rank_in_col) {
                    int index = (*sdispls)[target_proc] + current_offset[target_proc];
                    (*sendbuf_ids)[index] = id;
                    #pragma omp atomic
                    current_offset[target_proc]++;
                }
            }

        }
    }


    MPI_Alltoallv((*sendbuf_ids).data(), (*sendcounts).data(), (*sdispls).data(),
                  MPI_INT, (*receivebuf_ids).data(),
                  receive_counts_cyclic->data(), rdispls_cyclic->data(),
                  MPI_INT, grid->col_world);

    }

    unique_ptr<vector<DataTuple<DENT, embedding_dim>>> sendbuf_data =
        unique_ptr<vector<DataTuple<DENT, embedding_dim>>>(new vector<DataTuple<DENT, embedding_dim>>());

    for (int i = 0; i < grid->col_world_size; i++) {
        total_receive_count += (*receive_counts_cyclic)[i];
        total_send_count += (*sendcounts)[i];
    }

    sendbuf_data->resize(total_receive_count);

    unique_ptr<vector<DataTuple<DENT, embedding_dim>>> receivebuf_data =
        unique_ptr<vector<DataTuple<DENT, embedding_dim>>>(new vector<DataTuple<DENT, embedding_dim>>());
    receivebuf_data->resize(total_send_count);

    #pragma omp parallel for
    for (int j = 0; j < (*receivebuf_ids).size(); j++) {
        int local_key = (*receivebuf_ids)[j] - (grid->rank_in_col) * (this->sp_local_receiver)->proc_row_width;
        std::array<DENT, embedding_dim> val_arr = (this->dense_local)->fetch_local_data(local_key);
        (*sendbuf_data)[j].col = (*receivebuf_ids)[j];
        (*sendbuf_data)[j].value = val_arr;
    }


    MPI_Alltoallv((*sendbuf_data).data(), receive_counts_cyclic->data(), rdispls_cyclic->data(),
                  DENSETUPLE, receivebuf_data->data(),
                  (*sendcounts).data(), (*sdispls).data(),
                  DENSETUPLE, grid->col_world);


    MPI_Request dumy;

    this->populate_cache(sendbuf_data.get(), receivebuf_data.get(), &dumy, true, iteration, batch_id, true, false);
}



  inline void transfer_and_update_transpose(CSRLocal<SPT, DENT>* csr_local,CSRLocal<SPT, DENT>* csr_transpose,
                                            std::vector<SPT>& row_offsets_trans,std::vector<SPT>& col_indices_trans, std::vector<DENT>& values_trans){
    vector<int> send_counts(grid->col_world_size,0);
    vector<int> receive_counts(grid->col_world_size,0);

    vector<int> s_displs(grid->col_world_size,0);
    vector<int> r_displs(grid->col_world_size,0);

    CSRHandle<SPT,DENT> * csr_handle_local = csr_local->handler.get();
    CSRHandle<SPT,DENT> * csr_handle_transpose = csr_transpose->handler.get();

    std::vector<SPT>& row_offsets = csr_handle_local->rowStart;
    std::vector<SPT>& col_indices =  csr_handle_local->col_idx;
    std::vector<DENT>& values = csr_handle_local->values;

    std::vector<SPT>& transpose_row_offsets = csr_handle_transpose->rowStart;
    std::vector<SPT>& transpose_col_indices =  csr_handle_transpose->col_idx;
    std::vector<DENT>& transpose_values = csr_handle_transpose->values;


    int numRows = row_offsets.size()-1;
    int total_send_count=0;
    for(int i=0;i<numRows;i++){
      for(int j=row_offsets[i];j<row_offsets[i+1];j++){
          SPT column_id = col_indices[j];
          int target_rank = column_id/(this->sp_local_receiver)->proc_row_width;
          send_counts[target_rank]++;
          total_send_count++;
      }
    }

    int total_receive_count=0;
    for(int proc=0;proc<grid->col_world_size;proc++){
      int start_row_index = proc*(this->sp_local_receiver)->proc_row_width;
      int end_row_index = (proc+1)*(this->sp_local_receiver)->proc_row_width;
       if (proc == (grid->col_world_size-1)){
        end_row_index = (this->sp_local_receiver)->gRows;
      }
      receive_counts[proc]=transpose_row_offsets[end_row_index]-transpose_row_offsets[start_row_index];
      total_receive_count+=receive_counts[proc];
    }

    unique_ptr<vector<Tuple<DENT>>> send_value_ptr = make_unique<vector<Tuple<DENT>>>(total_send_count);
    unique_ptr<vector<Tuple<DENT>>> receive_value_ptr = make_unique<vector<Tuple<DENT>>>(total_receive_count);


    for(int proc=0;proc<grid->col_world_size;proc++){
      s_displs[proc]= proc>0?(s_displs[proc-1]+send_counts[proc-1]):s_displs[proc];
      r_displs[proc]= proc>0?(r_displs[proc-1]+receive_counts[proc-1]):r_displs[proc];
    }

    vector<int> offsets(grid->col_world_size,0);
    for(int i=0;i<numRows;i++){
      for(int j=row_offsets[i];j<row_offsets[i+1];j++){
        SPT column_id = col_indices[j];
        int target_rank = column_id/(this->sp_local_receiver)->proc_row_width;
        int index =  offsets[target_rank] + s_displs[target_rank];
        (*send_value_ptr)[index].value = values[j];
        (*send_value_ptr)[index].col = i + grid->rank_in_col*(this->sp_local_receiver)->proc_row_width;
        (*send_value_ptr)[index].row = column_id - target_rank*(this->sp_local_receiver)->proc_row_width;
        offsets[target_rank]++;
      }
    }

    MPI_Alltoallv((*send_value_ptr).data(), (send_counts).data(), (s_displs).data(),
                  SPTUPLE, (*receive_value_ptr.get()).data(),
                  (receive_counts).data(), (r_displs).data(),
                  SPTUPLE, grid->col_world);

    unique_ptr<CSRLocal<SPT,DENT>> csr_transpose_ptr = make_unique<CSRLocal<SPT,DENT>>((numRows),static_cast<int>((this->sp_local_receiver)->gRows),
                                                                                         (total_receive_count),(*receive_value_ptr.get()).data(),total_receive_count,false);

    row_offsets_trans = csr_transpose_ptr.get()->handler.get()->rowStart;
    col_indices_trans = csr_transpose_ptr.get()->handler.get()->col_idx;
    values_trans = csr_transpose_ptr.get()->handler.get()->values;
    cout<<" rank "<<grid->rank_in_col<<" nnz  after copying "<<row_offsets_trans[row_offsets_trans.size()-1]<<endl;

  }


  inline void populate_cache(std::vector<DataTuple<DENT, embedding_dim>> *sendbuf,
                             std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                      MPI_Request *req, bool synchronous, int iteration,
                      int batch_id, bool temp, bool clear_buff=true) {
    if (!synchronous) {
      MPI_Status status;
      auto t = start_clock();
      MPI_Wait(req, &status);
      stop_clock_and_add(t, "Embedding Communication Time");
    }
//    (this->dense_local)->invalidate_cache(iteration,batch_id,temp);
    for (int i = 0; i < this->grid->col_world_size; i++) {
      auto base_index = (*rdispls_cyclic)[i];
      auto count = (*receive_counts_cyclic)[i];
 //     (*rdispls_cyclic)[i]=0;
 //     (*receive_counts_cyclic)[i]=0;
      for (auto j = base_index; j < base_index + count; j++) {
        DataTuple<DENT, embedding_dim> t = (*receivebuf)[j];
        (this->dense_local)->insert_cache(i, t.col, batch_id, iteration, t.value, temp);
      }
    }
    if(clear_buff) {
      receivebuf->clear();
      receivebuf->shrink_to_fit();
      sendbuf->clear();
      sendbuf->shrink_to_fit();
    }
  }


};

} // namespace distblas::net
