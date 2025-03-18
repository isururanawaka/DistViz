#pragma once
#include "../net/process_3D_grid.hpp"
#include "../common/common.h"
#include "distributed_mat.hpp"
#include "sparse_mat.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>
#include "../io/file_writer.hpp"

using namespace std;
using namespace Eigen;
using namespace hipgraph::distviz::net;
using namespace hipgraph::distviz::common;
using namespace hipgraph::distviz::io;

namespace hipgraph::distviz::embedding {

/**
 * This class represents  the dense matrix.
 */
template <typename SPT, typename DENT, size_t embedding_dim>
class DenseMat : DistributedMat {

private:
public:
  uint64_t rows;
  unique_ptr<vector<unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>>>
      cachePtr;
  unique_ptr<vector<unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>>>
      tempCachePtr;
  DENT *nCoordinates;
  Process3DGrid *grid;

  /**
   *
   * @param rows Number of rows of the matrix
   * @param cols  Number of cols of the matrix
   * @param init_mean  initialize with normal distribution with given mean
   * @param std  initialize with normal distribution with given standard
   * deviation
   */
  DenseMat(Process3DGrid *grid, uint64_t rows,bool initialize=true) {

    this->rows = rows;
    this->grid = grid;

    cout<<" col world size "<<grid->col_world_size<<endl;

    this->cachePtr = std::make_unique<std::vector<
        std::unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>>>(
        grid->col_world_size);
    this->tempCachePtr = std::make_unique<std::vector<
        std::unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>>>(
        grid->col_world_size);
    nCoordinates =
        static_cast<DENT *>(::operator new(sizeof(DENT[rows * embedding_dim])));
    uint64_t seed=0;
    if (grid->rank_in_col==0){
      std::random_device rd;
      seed  = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, grid->col_world);

    std::mt19937_64 gen(seed);
    //    gen.seed(seed);
    int subset_start = rows* grid->rank_in_col*embedding_dim;
    gen.discard(subset_start);

    std::uniform_int_distribution<int> dist(0, RAND_MAX);

//    std::srand(this->grid->global_rank);
    if (initialize) {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < embedding_dim; j++) {
          DENT val = -1.0 + 2.0 * dist(gen) / (RAND_MAX + 1.0);
          nCoordinates[i * embedding_dim + j] = val;
        }
      }
    }
  }

  ~DenseMat() {}

  void insert_cache(int rank, uint64_t key, int batch_id, int iteration,
                    std::array<DENT, embedding_dim> &arr, bool temp) {
    CacheEntry<DENT, embedding_dim> entry;
    entry.inserted_batch_id = batch_id;
    entry.inserted_itr = iteration;
    entry.value = arr;
    if (temp) {
      (*this->tempCachePtr)[rank][key]= entry;
    } else {
      (*this->cachePtr)[rank][key]= entry;
    }
  }

  auto fetch_data_vector_from_cache(std::array<DENT, embedding_dim> &value,int rank, uint64_t key, bool temp) {

    // Access the array using the provided rank and key

    auto arrayMap = (temp) ? (*tempCachePtr)[rank] : (*cachePtr)[rank];
    auto it = arrayMap.find(key);

    if (it != arrayMap.end()) {
      auto temp = it->second;
      value =  temp.value;
    }else {
      throw std::runtime_error("cannot find the given key");
    }

  }

  std::array<DENT, embedding_dim> fetch_local_data(int local_key) {
    std::array<DENT, embedding_dim> stdArray;

    int base_index = local_key * embedding_dim;
    std::copy(nCoordinates + base_index,
              nCoordinates + base_index + embedding_dim, stdArray.data());
    return stdArray;
  }

  void invalidate_cache(int current_itr, int current_batch, bool temp) {
    if (temp) {
      purge_temp_cache();
    } else {
      for (int i = 0; i < grid->col_world_size; i++) {
        auto &arrayMap = (*cachePtr)[i];
        for (auto it = arrayMap.begin(); it != arrayMap.end();) {
          CacheEntry<DENT, embedding_dim> cache_ent =
              it->second;
          if (cache_ent.inserted_itr < current_itr and
              cache_ent.inserted_batch_id <= current_batch) {
            it = arrayMap.erase(it);
          } else {
            // Move to the next item
            ++it;
          }
        }
      }
    }
  }

  void purge_temp_cache() {
    for (int i = 0; i < grid->col_world_size; i++) {
      (*this->tempCachePtr)[i].clear();
      std::unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>().swap(
          (*this->tempCachePtr)[i]);
    }
  }

  // Utitly methods
  void print_matrix() {
    int rank = grid->rank_in_col;
    string output_path = "embedding" + to_string(rank) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);
    for (int i = 0; i < rows; ++i) {
      fout << (i + 1) << " ";
      for (int j = 0; j < embedding_dim; ++j) {
        fout << nCoordinates[i * embedding_dim + j] << " ";
      }
      fout << endl;
    }
  }

  void print_matrix_rowptr(int iter) {
    int rank= grid->rank_in_col;
    string output_path =to_string(iter) + "_embedding.txt";
//    char stats[500];
//    strcpy(stats, output_path.c_str());
//    ofstream fout(stats, std::ios_base::app);
//    //    fout << (*this->matrixPtr).rows() << " " << (*this->matrixPtr).cols()
//    //         << endl;
//    for (int i = 0; i < rows; ++i) {
//      fout << i  + rank * rows +1 << " ";
//      for (int j = 0; j < embedding_dim; ++j) {
//        fout << this->nCoordinates[i * embedding_dim + j] << " ";
//      }
//      fout << endl;
//    }

    FileWriter<int,float,embedding_dim> fileWriter;
    fileWriter.parallel_write(output_path,this->nCoordinates,rows, embedding_dim);
  }

  void print_cache(int iter) {
    int rank = grid->rank_in_col;
    cout<<" rank "<<rank<<" cache  size "<<(*this->tempCachePtr).size()<<endl;
//    for (int i = 0; i < (*this->tempCachePtr).size(); i++) {
//      unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>& map =
////          (*this->cachePtr)[i];
//      (*this->tempCachePtr)[i];


      string output_path = "cache.txt";
//      char stats[500];
//      strcpy(stats, output_path.c_str());
//      ofstream fout(stats, std::ios_base::app);
//
//      int size = map.size();
//      int total_st = size*embedding_dim;
//
//      DENT*  coord[size*embedding_dim];
//
//      for (const auto &kvp : map) {
//        uint64_t key = kvp.first;
//        const std::array<DENT, embedding_dim> &value = kvp.second.value;
//        fout << (key +1)<< " ";
//        for (int i = 0; i < embedding_dim; ++i) {
//          fout << value[i] << " ";
//        }
////        fout << std::endl;
//      }
      FileWriter<int,float,embedding_dim> fileWriter;
      fileWriter.parallel_write_map(output_path,this->tempCachePtr.get());
//    }
  }

  bool searchForKey(uint64_t key) {
    for (const auto &nestedMap : *cachePtr) {
      auto it = nestedMap.find(key);
      if (it != nestedMap.end()) {
        auto result = it->second;
        return true; // Key found in the current nestedMap
      }
    }
    return false; // Key not found in any nestedMap
  }
};

} // namespace distblas::core
