/**
 * This class implements the distributed sparse graph.
 */
#pragma once
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <parallel/algorithm>
#include <unordered_set>
#include <vector>
#include "../common/common.h"
#include "../net/process_3D_grid.hpp"

using namespace std;
using namespace hipgraph::distviz::net;

namespace hipgraph::distviz::embedding {

/**
 * This class represents the Sparse Matrix
 */
template <typename INDEX_TYPE,typename  VALUE_TYPE> class SpMat : public DistributedMat {

private:
public:
  uint64_t gRows, gCols, gNNz;
  vector<Tuple<VALUE_TYPE>>* coords;
  int batch_size;
  int proc_col_width, proc_row_width;
  bool transpose = false;
  bool col_partitioned = false;
  unique_ptr<CSRLocal<INDEX_TYPE,VALUE_TYPE>> csr_local_data;
  Process3DGrid *grid;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(Process3DGrid* grid, vector<Tuple<VALUE_TYPE>>* coords, uint64_t &gRows, uint64_t &gCols,
        uint64_t &gNNz, int &batch_size, int &proc_row_width,
        int &proc_col_width, bool transpose, bool col_partitioned) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
    this->batch_size = batch_size;
    this->proc_col_width = proc_col_width;
    this->proc_row_width = proc_row_width;
    this->transpose = transpose;
    this->col_partitioned = col_partitioned;
    this->grid = grid;
  }

  SpMat(Process3DGrid *grid) {
    this->grid = grid;
  }

  /**
   * Initialize the CSR from (*coords) data structure
   */
  void initialize_CSR_blocks(bool sort=false) {
    cout<<" rank "<<this->grid->rank_in_col<<" trying to initialize "<<endl;

    if (sort){
      if (col_partitioned){
        std::sort((*coords).begin(), (*coords).end(),column_major<VALUE_TYPE>);
      } else {
        std::sort((*coords).begin(), (*coords).end(),row_major<VALUE_TYPE>);
      }
    }
     #pragma omp parallel for
    for (uint64_t i = 0; i < (*coords).size(); i++) {
      if (col_partitioned) {
        (*coords)[i].col %= proc_col_width;
      } else {
        (*coords)[i].row %= proc_row_width;
//        if ((*coords)[i].col>= this->gCols) {
//            cout<<" rank "<<this->grid->rank_in_col<<" col index outdated  "<<(*coords)[i].col<<endl;
//        }
      }
    }

    cout<<" rank "<<this->grid->rank_in_col<<" data filling completed "<<endl;

    Tuple<VALUE_TYPE> *coords_ptr = (*coords).data();
    if (col_partitioned) {
      // This is used to find sending indices
      csr_local_data =
          make_unique<CSRLocal<INDEX_TYPE,VALUE_TYPE>>(gRows, proc_col_width, (*coords).size(),
                                   coords_ptr, (*coords).size(), transpose);
    } else {
      // This is used to find receiving indices and computations

      csr_local_data =
          make_unique<CSRLocal<INDEX_TYPE,VALUE_TYPE>>(proc_row_width, gCols, (*coords).size(), coords_ptr, (*coords).size(), transpose);
    }
  }

  // if batch_id<0 it will fetch all the batches
  void fill_col_ids(int batch_id, int starting_proc, int end_proc, vector<unordered_set<uint64_t>>*proc_to_id_mapping,
                    unordered_map<uint64_t, unordered_map<int,bool>> *id_to_proc_mapping, bool mode) {

    if (mode == 0) {

      fill_col_ids_for_pulling(batch_id,starting_proc,end_proc, proc_to_id_mapping,id_to_proc_mapping);
    } else {
      fill_col_ids_for_pushing(batch_id, starting_proc,end_proc,proc_to_id_mapping,id_to_proc_mapping);
    }
  }

  /*
   * This method computes all indicies for pull based approach
   */
  void fill_col_ids_for_pulling(int batch_id, int starting_proc, int end_proc, vector<unordered_set<uint64_t>> *proc_to_id_mapping,
                                unordered_map<uint64_t, unordered_map<int,bool>> *id_to_proc_mapping) {

    int rank= grid->rank_in_col;
    int world_size = grid->col_world_size;

    CSRHandle<INDEX_TYPE,VALUE_TYPE> *handle = (csr_local_data.get())->handler.get();

    vector<int> procs;
    for (int i = starting_proc; i < end_proc; i++) {
      int  target   = (col_partitioned)? (rank + i) % world_size: (rank >= i) ? (rank - i) % world_size : (world_size - i + rank) % world_size;
      procs.push_back(target);
    }

    ofstream fout;
    fout.open("/global/homes/i/isjarana/distviz_executions/perf_comparison/DistViz/MNIST/acesss_ids.txt", std::ios_base::app);
    if (col_partitioned) {
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index = batch_id * batch_size + proc_row_width * procs[r];
        auto end_index =  std::min(std::min((starting_index+batch_size),static_cast<uint64_t>((procs[r] + 1) * proc_row_width)), gRows);
        for (auto i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];j++) {
              auto col_val = handle->col_idx[j];
             // fout<<(i+1)<<" "<<(col_val+1)+proc_row_width * rank<<" "<< handle->values[j]<<endl;
              { (*proc_to_id_mapping)[procs[r]].insert(col_val);
                (*id_to_proc_mapping)[col_val][procs[r]] = true;
              }
            }
          }
        }
      }
    } else if (transpose) {
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index = proc_col_width * procs[r];
        auto end_index =
            std::min(static_cast<uint64_t>((procs[r] + 1) * proc_col_width), gCols);
        for (auto i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1]; j++) {
              auto col_val = handle->col_idx[j];
              uint64_t dst_start = batch_id * batch_size;
              uint64_t dst_end_index = std::min((batch_id + 1) * batch_size, proc_row_width);
              if (col_val >= dst_start and col_val < dst_end_index) {
                { (*proc_to_id_mapping)[procs[r]].insert(i);
                }
              }
            }
          }
        }
      }
    }
  }

  /*
   * This method computes all indicies for push based approach
   */
  void fill_col_ids_for_pushing(int batch_id,int starting_proc, int end_proc, vector<unordered_set<uint64_t>>* proc_to_id_mapping,
                                unordered_map<uint64_t, unordered_map<int,bool>>* id_to_proc_mapping) {
    int rank= grid->rank_in_col;
    int world_size = grid->col_world_size;

    CSRHandle<INDEX_TYPE,VALUE_TYPE> *handle = (csr_local_data.get())->handler.get();

    auto batches = (proc_row_width / batch_size);

    if (!(proc_row_width % batch_size == 0)) {
      batches = (proc_row_width / batch_size) + 1;
    }

    vector<int> procs;
    for (int i = starting_proc; i < end_proc; i++) {
      int  target  = (col_partitioned)? (rank + i) % world_size: (rank >= i) ? (rank - i) % world_size : (world_size - i + rank) % world_size;
      procs.push_back(target);
    }

    if (col_partitioned) {
      // calculation of sender col_ids
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index = proc_row_width * procs[r];
        auto end_index = std::min(static_cast<uint64_t>((procs[r] + 1) * proc_row_width), gRows) -1;

        auto eligible_col_id_start =
            (batch_id >= 0) ? batch_id * batch_size : 0;
        auto eligible_col_id_end =
            (batch_id >= 0)
                ? std::min(static_cast<uint64_t>((batch_id + 1) * batch_size),
                           static_cast<uint64_t>(proc_col_width))
                : proc_col_width;

        for (auto i = starting_index; i <= (end_index); i++) {

          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              if (col_val >= eligible_col_id_start and
                  col_val < eligible_col_id_end) {
                // calculation of sender col_ids
                { (*proc_to_id_mapping)[procs[r]].insert(col_val);
                  (*id_to_proc_mapping)[col_val][procs[r]] = true;
                }
              }
            }
          }
        }
      }
    } else if (transpose) {
      // calculation of receiver col_ids
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index =
            (batch_id >= 0) ? batch_id * batch_size + proc_col_width * procs[r]
                            : proc_col_width *  procs[r];
        auto end_index =
            (batch_id >= 0)
                ? std::min(
                      starting_index + batch_size,
                      std::min(static_cast<uint64_t>(( procs[r] + 1) * proc_col_width),
                               gCols)) -
                      1
                : std::min(static_cast<uint64_t>(( procs[r] + 1) * proc_col_width),
                           gCols) -
                      1;
        for (auto i = starting_index; i <= (end_index); i++) {
          if (rank !=  procs[r] and (handle->rowStart[i + 1] - handle->rowStart[i]) > 0 ) {
            (*proc_to_id_mapping)[procs[r]].insert(i);
          }
        }
      }
    }
  }

  void print_coords(bool trans) {
    int rank= grid->rank_in_col;
    int world_size = grid->col_world_size;
    string output_path =
        "coords" + to_string(rank) + "trans" + to_string(trans) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);

    for (int i = 0; i < (*coords).size(); i++) {
      fout << (*coords)[i].row << " " << (*coords)[i].value << " " << endl;
    }
  }

  void print_csr() {
    int rank= grid->rank_in_col;
    int world_size = grid->col_world_size;
    string output_path =
        "coords" + to_string(rank) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);

    CSRHandle<INDEX_TYPE,VALUE_TYPE>* handle = csr_local_data.get()->handler.get();
    cout<<"  rank "<<rank<<" csr size "<<(*handle).rowStart.size()<<endl;
    for(auto i=0;i<((*handle).rowStart.size()-1);i++){
      fout<<i<<" ";
      for(auto j=(*handle).rowStart[i];j<(*handle).rowStart[i+1];j++){
        fout<<(*handle).col_idx[j]<<" ";
      }
      fout<<endl;
    }

  }

  ~SpMat() {}
};

} // namespace distblas::core
