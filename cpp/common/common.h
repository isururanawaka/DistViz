#ifndef DISTVIZ_CPP_CORE_COMMON_H_
#define DISTVIZ_CPP_CORE_COMMON_H_

#define MPI_VALUE_TYPE MPI_FLOAT
#define MPI_INDEX_TYPE MPI_INT

#include <vector>
#include "mpi_type_creator.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <cstdint> // int64_t
#include <iostream>
#include <mkl_spblas.h>
#include <mpi.h>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include "json.hpp"
#include <unordered_map>
#include <unistd.h>
#include <unordered_set>
#include <algorithm>


using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t;

namespace hipgraph::distviz::common {

template <typename INDEX_TYPE>
struct index_distance_pair
{
  float distance;
  INDEX_TYPE index;
};

struct LeafPriority {
  int leaf_index;
  float priority;
};

template <typename INDEX_TYPE, typename VALUE_TYPE>
struct DataNode {
  INDEX_TYPE index;
  VALUE_TYPE value;
  vector<VALUE_TYPE> image_data;
};

template <typename INDEX_TYPE, typename VALUE_TYPE>
struct  EdgeNode {
  INDEX_TYPE src_index;
  INDEX_TYPE dst_index;
  VALUE_TYPE distance;
};

template<typename T>
vector <T> slice (vector<T> const &v, int m, int n) {
  auto first = v.cbegin () + m;
  auto last = v.cbegin () + n + 1;
  std::vector <T> vec (first, last);
  return vec;
}

enum class StorageFormat { RAW, COLUMN };

template<class T, class X>
void sortByFreq (std::vector<T> &v, std::vector<X> &vec, int world_size)
{
  std::unordered_map <T, size_t> count;

  for (int i=0; i< v.size();i++)
  {
    count[v[i]]++;
  }

  std::sort (v.begin (), v.end (), [&count] (T const &a, T const &b)
            {
              if (a == b)
              {
                return false;
              }
              if (count[a] > count[b])
              {
                return true;
              }
              else if (count[a] < count[b])
              {
                return false;
              }
              return a < b;
            });
  auto last = std::unique (v.begin (), v.end ());
  v.erase (last, v.end ());

  for (int i=0; i< v.size();i++)
  {
    float priority = static_cast<float>(count[v[i]] / world_size);

    hipgraph::distviz::common::LeafPriority leaf_priority;
    leaf_priority.priority = priority;
    leaf_priority.leaf_index = v[i];
    vec[v[i]] = leaf_priority;
  }
}

template<typename T>
bool all_equal(std::vector<T> const& v)
{
  return std::adjacent_find(v.begin(), v.end(), std::not_equal_to<T>()) == v.end();
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
using DataNode3DVector = vector<vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE>>>>;

template <typename VALUE_TYPE>
using ValueType2DVector = vector<vector<VALUE_TYPE>>;

int divide_and_round_up(uint64_t num, int denom);
template <typename INDEX_TYPE>
vector<INDEX_TYPE> generate_random_numbers(int lower_bound,int upper_bound,int seed, int ns) {
  vector<INDEX_TYPE> vec(ns);
  std::minstd_rand generator(seed);

  // Define the range of the uniform distribution
  std::uniform_int_distribution<int> distribution(lower_bound, upper_bound-1);

  // Generate and print random numbers
  //#pragma omp parallel
  for (int i = 0; i < ns; ++i) {
    int random_number = distribution(generator);
    vec[i] = static_cast<INDEX_TYPE>(random_number);
  }
  return vec;
}



void prefix_sum(vector<int> &values, vector<int> &offsets);

size_t get_memory_usage();

uint64_t  tau_rand_int(std::array<uint64_t,4> shuffle_table);



void initialize_shuffle_table(std::array<uint64_t,4>& shuffle_table);

uint64_t xorshift64(uint64_t* state);

void reset_performance_timers();

void stop_clock_and_add(my_timer_t &start, string counter_name);

void add_memory(size_t mem, string counter_name);

void add_datatransfers(uint64_t count, string counter_name);

void print_performance_statistics();

my_timer_t start_clock();

double stop_clock_get_elapsed(my_timer_t &start);

json json_perf_statistics();

int get_proc_length(double beta, int world_size);

int get_end_proc(int  starting_index, double beta, int world_size);

std::unordered_set<uint64_t> random_select(const std::unordered_set<uint64_t>& originalSet, int count);




template <typename T>
struct Tuple {
    int64_t row;
    int64_t col;
    T value;

    // Default Constructor
    Tuple() = default;

    // Parameterized Constructor
    Tuple(int64_t r, int64_t c, T v) : row(r), col(c), value(v) {}

    // Copy Constructor
    Tuple(const Tuple<T>& other)
            : row(other.row), col(other.col), value(other.value) {}

    // Copy Assignment Operator
    Tuple<T>& operator=(const Tuple<T>& other) {
        if (this != &other) {
            row = other.row;
            col = other.col;
            value = other.value;
        }
        return *this;
    }

    // Move Constructor
    Tuple(Tuple<T>&& other) noexcept
            : row(other.row), col(other.col), value(std::move(other.value)) {}

    // Move Assignment Operator
    Tuple<T>& operator=(Tuple<T>&& other) noexcept {
        if (this != &other) {
            row = other.row;
            col = other.col;
            value = std::move(other.value);
        }
        return *this;
    }
};


template <typename T> struct CSR {
  int64_t row;
  int64_t col;
  T value;
};

template <typename T, size_t size> struct DataTuple {
  uint64_t col;
  std::array<T, size> value;
};

template <typename T, size_t size> struct CacheEntry {
  std::array<T, size> value;
  int inserted_batch_id;
  int inserted_itr;
};

template <typename  INDEX_TYPE, typename VALUE_TYPE>
struct CSRHandle {
  vector<VALUE_TYPE> values;
  vector<INDEX_TYPE> col_idx;
  vector<INDEX_TYPE> rowStart;
  vector<INDEX_TYPE> row_idx;
};



template <typename T>
bool CompareTuple(const Tuple<T>& obj1, const Tuple<T>& obj2) {
  // Customize the comparison logic based on your requirements
  return obj1.col == obj2.col;
}

// TODO: removed reference type due to binding issue
template <typename T> bool column_major(Tuple<T>& a, Tuple<T>& b) {
  if (a.col == b.col) {
    return a.row < b.row;
  } else {
    return a.col < b.col;
  }
}

template <typename T> bool row_major(Tuple<T>& a, Tuple<T>& b) {
  if (a.row == b.row) {
    return a.col < b.col;
  } else {
    return a.row < b.row;
  }
}

extern MPI_Datatype SPTUPLE;

extern MPI_Datatype DENSETUPLE;


extern vector<string> perf_counter_keys;

extern map<string, int> call_count;
extern map<string, double> total_time;

template <typename T> void initialize_mpi_datatype_SPTUPLE() {
  const int nitems = 3;
  int blocklengths[3] = {1, 1, 1};
  MPI_Datatype *types = new MPI_Datatype[3];
  types[0] = MPI_UINT64_T;
  types[1] = MPI_UINT64_T;
  MPI_Aint offsets[3];
  if (std::is_same<T, int>::value) {
    types[2] = MPI_INT;
    offsets[0] = offsetof(Tuple<int>, row);
    offsets[1] = offsetof(Tuple<int>, col);
    offsets[2] = offsetof(Tuple<int>, value);
  } else {
    // TODO:Need to support all datatypes
    types[2] = MPI_DOUBLE;
    offsets[0] = offsetof(Tuple<double>, row);
    offsets[1] = offsetof(Tuple<double>, col);
    offsets[2] = offsetof(Tuple<double>, value);
  }

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &SPTUPLE);
  MPI_Type_commit(&SPTUPLE);
  delete[] types;
}

template <typename T,size_t embedding_dim>
void initialize_mpi_datatype_DENSETUPLE() {
  DataTuple<T,embedding_dim> p;
  DENSETUPLE = CreateCustomMpiType(p, p.col, p.value);
}

template <typename SPT, typename DENT, size_t embedding_dim>
void initialize_mpi_datatypes() {
  initialize_mpi_datatype_SPTUPLE<SPT>();
  initialize_mpi_datatype_DENSETUPLE<DENT,embedding_dim>();
}

template <typename DENT, size_t MAXBOUND>
DENT  scale(DENT v){
  if(v > MAXBOUND) return MAXBOUND;
  else if(v < -MAXBOUND) return -MAXBOUND;
  else return v;
}

}



#endif // DISTVIZ_CPP_CORE_COMMON_H_
