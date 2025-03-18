#pragma once
#include "Eigen/Dense"
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
#include <string>
#include <fstream>
#include <iostream>
#include "../common/common.h"
#include "../net/process_3D_grid.hpp"
#include "CombBLAS/CombBLAS.h"
#include "../embedding/sparse_mat.hpp"



using namespace std;
using namespace hipgraph::distviz::common;
using namespace hipgraph::distviz::net;
using namespace  hipgraph::distviz::embedding;
using namespace combblas;
namespace hipgraph::distviz::io {




template <typename INDEX_TYPE,typename VALUE_TYPE, size_t DIM>
class FileReader {

public:
 static Eigen::MatrixXf load_data(string file_path, int no_of_images,
                                   int dimension, int rank, int world_size) {
   // Read the file
   std::ifstream file(file_path, std::ios::binary);

   if (!file.is_open()) {
     std::cout << "Error: Could not open the file." << std::endl;
     return Eigen::MatrixXf();
   }

   int magic_number = 0;
   int number_of_images = 0;
   int n_rows = 0;
   int n_cols = 0;
   file.read((char *)&magic_number, sizeof(magic_number));
   magic_number = reverse_int(magic_number);
   file.read((char *)&number_of_images, sizeof(number_of_images));
   number_of_images = reverse_int(number_of_images);
   file.read((char *)&n_rows, sizeof(n_rows));
   n_rows = reverse_int(n_rows);
   file.read((char *)&n_cols, sizeof(n_cols));
   n_cols = reverse_int(n_cols);


   int chunk_size = number_of_images / world_size;
   if (rank == world_size - 1) {
     chunk_size = no_of_images - chunk_size * (world_size - 1);
   }
   Eigen::MatrixXf matrix(chunk_size, dimension);

   for (int i = 0; i < number_of_images; ++i) {
     for (int r = 0; r < n_rows; ++r) {
       for (int c = 0; c < n_cols; ++c) {
         unsigned char temp = 0;
         file.read((char *)&temp, sizeof(temp));
         if (i >= rank * chunk_size and i < (rank + 1) * chunk_size and rank < world_size - 1) {
           matrix(i - rank * chunk_size,(n_rows * r) + c) = (float)temp;
         } else if (rank == world_size - 1 && i >= (rank)*chunk_size) {
           matrix(i - rank * chunk_size,(n_rows * r) + c) = (float)temp;
         }
       }
     }
   }
   file.close();

   // Create Eigen::MatrixXf
   return matrix;
 }

static void ubyte_read(string file_path,ValueType2DVector<VALUE_TYPE>* datamatrix,
                                      int no_of_datapoints,int dimension, int rank, int world_size){

  ifstream file (file_path, ios::binary);
  if (file.is_open ())
  {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read ((char *) &magic_number, sizeof (magic_number));
    magic_number = reverse_int (magic_number);
    file.read ((char *) &number_of_images, sizeof (number_of_images));
    number_of_images = reverse_int (number_of_images);
    file.read ((char *) &n_rows, sizeof (n_rows));
    n_rows = reverse_int (n_rows);
    file.read ((char *) &n_cols, sizeof (n_cols));
    n_cols = reverse_int (n_cols);

    int chunk_size = number_of_images / world_size;
    if (rank < world_size - 1)
    {
      datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    }
    else if (rank == world_size - 1)
    {
      chunk_size = no_of_datapoints - chunk_size * (world_size - 1);
      datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    }

    for (int i = 0; i < number_of_images; ++i)
    {
      for (int r = 0; r < n_rows; ++r)
      {
        for (int c = 0; c < n_cols; ++c)
        {
          unsigned char temp = 0;
          file.read ((char *) &temp, sizeof (temp));
          if (i >= rank * chunk_size and i < (rank + 1) * chunk_size and rank < world_size - 1)
          {
            (*datamatrix)[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE) temp;
          }
          else if (rank == world_size - 1 && i >= (rank) * chunk_size)
          {
            (*datamatrix)[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE) temp;
          }
        }
      }
    }
  }
  file.close ();
}

static void fvecs_read(string filename, ValueType2DVector<VALUE_TYPE>* datamatrix,
                int no_of_datapoints,int dimension, int rank, int world_size) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "I/O error: Unable to open the file " << filename << "\n";
    exit(EXIT_FAILURE);
  }

  // Read the vector size
  int d;
  file.read(reinterpret_cast<char*>(&d), sizeof(int));
  int vecsizeof = 1 * 4 + d * 4;

  int chunk_size = no_of_datapoints / world_size;

  vector<int>bounds = vector<int>(2);
  if (rank < world_size - 1){
    datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    bounds[0]= rank * chunk_size;
    bounds[1] = (rank+1) * chunk_size -1;
  }else if (rank == world_size - 1){
    chunk_size = no_of_datapoints - chunk_size * (world_size - 1);
    datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    bounds[0]= rank * chunk_size;
    bounds[1] = std::min((rank+1) * chunk_size -1,no_of_datapoints-1);
  }

  cout<<" rank "<<rank<<" a "<<bounds[0]<<" b "<<bounds[1]<<"d"<<d<<endl;

  // Get the number of vectors
  file.seekg(0, std::ios::end);
  int a = 0;  // Start from 0-based index
  int bmax = file.tellg() / vecsizeof;
  int b = bmax - 1;  // Convert to 0-based index

  if (!bounds.empty()) {
    if (bounds.size() == 1) {
      b = bounds[0] ;  // Convert to 0-based index
    } else if (bounds.size() == 2) {
      a = bounds[0];
      b = bounds[1];  // Convert to 0-based index
    }
  }

  assert(a >= 0);

  if (b == -1 || b < a) {
    return;  // Return an empty vector
  }

  // Compute the number of vectors that are really read and go to starting positions
  int n = b - a + 1;
  file.seekg(a * vecsizeof, std::ios::beg);  // Adjust to 0-based index

  // Read n vectors
  std::vector<float> rawVectors((d + 1) * n);
  file.read(reinterpret_cast<char*>(rawVectors.data()), sizeof(float) * (d + 1) * n);

  // Check if the first column (dimension of the vectors) is correct
//  assert(std::count(rawVectors.begin() + 1, rawVectors.end(), rawVectors[0]) == n - 1);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      (*datamatrix)[i][j] = rawVectors[(d + 1) * i + j + 1];
    }
  }
}

static void  read_fbin(string filename, ValueType2DVector<VALUE_TYPE>* datamatrix,
                      INDEX_TYPE no_of_datapoints,int dim, int rank, int world_size, INDEX_TYPE offset=8) {
  cout<<" rank  "<<rank<<"  openinig file "<<filename<<endl;
  std::ifstream file(filename, std::ios::binary);
  cout<<" rank  "<<rank<<"  openinig file "<<filename<<endl;
  if (!file.is_open()) {
    // Handle file opening error
    std::cerr << "Error: Unable to open the file " << filename << std::endl;
    return;
  }
  cout<<" rank  "<<rank<<"  openinig completed "<<filename<<endl;
  int nvecs=no_of_datapoints;

//  file.read(reinterpret_cast<char*>(&nvecs), sizeof(int));
//  file.read(reinterpret_cast<char*>(&dim), sizeof(int));
//
//  cout<<" rank  "<<rank<<"  nvecs "<<nvecs<<" dim "<<dim<<endl;

  INDEX_TYPE chunk_size = no_of_datapoints / world_size;
  INDEX_TYPE start_idx =rank*chunk_size;
  INDEX_TYPE end_index = 0;
  if (rank < world_size - 1){
    end_index = (rank+1) * chunk_size -1;
  }else if (rank == world_size - 1){
    end_index = std::min((rank+1) * chunk_size -1,no_of_datapoints-1);
    chunk_size = no_of_datapoints-(rank)*chunk_size;
  }

  if (chunk_size == -1) {
    chunk_size = no_of_datapoints - start_idx;
  }
  datamatrix->resize(chunk_size, vector<VALUE_TYPE> (dim));
  cout<<" rank  "<<rank<<"  selected chunk size  "<<chunk_size<<" starting "<<start_idx<<endl;
  std::vector<float> data(chunk_size * dim);

  file.seekg(start_idx * 4 * dim+offset, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), sizeof(float) * chunk_size * dim);
  const float scaleParameter = 10;
  cout<<" rank  "<<rank<<"  data reading  completed"<<endl;

  for (INDEX_TYPE i = 0; i < chunk_size; ++i) {
    std::vector<float> vec(dim);
    std::copy(data.begin() + i * dim, data.begin() + (i + 1) * dim, vec.begin());
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [scaleParameter](float value) { return value * scaleParameter; });

    (*datamatrix)[i]=vec;
  }
  cout<<" rank  "<<rank<<"  data loading completed"<<endl;
}

static void read_ubin(string filename, ValueType2DVector<VALUE_TYPE>* datamatrix,
                       INDEX_TYPE no_of_datapoints,int dim, int rank, int world_size, INDEX_TYPE offset=8) {
  cout<<" rank  "<<rank<<"  openinig file "<<filename<<endl;
  std::ifstream file(filename, std::ios::binary);
  cout<<" rank  "<<rank<<"  openinig file "<<filename<<endl;
  if (!file.is_open()) {
    // Handle file opening error
    std::cerr << "Error: Unable to open the file " << filename << std::endl;
    return;
  }
  cout<<" rank  "<<rank<<"  openinig completed "<<filename<<endl;
  int nvecs=no_of_datapoints;

  //  file.read(reinterpret_cast<char*>(&nvecs), sizeof(int));
  //  file.read(reinterpret_cast<char*>(&dim), sizeof(int));
  //
  //  cout<<" rank  "<<rank<<"  nvecs "<<nvecs<<" dim "<<dim<<endl;

  INDEX_TYPE chunk_size = no_of_datapoints / world_size;
  INDEX_TYPE start_idx =rank*chunk_size;
  INDEX_TYPE end_index = 0;
  if (rank < world_size - 1){
    end_index = (rank+1) * chunk_size -1;
  }else if (rank == world_size - 1){
    end_index = std::min((rank+1) * chunk_size -1,no_of_datapoints-1);
    chunk_size = no_of_datapoints-(rank)*chunk_size;
  }

  if (chunk_size == -1) {
    chunk_size = no_of_datapoints - start_idx;
  }
  datamatrix->resize(chunk_size, vector<VALUE_TYPE> (dim));
  cout<<" rank  "<<rank<<"  selected chunk size  "<<chunk_size<<" starting "<<start_idx<<endl;
  std::vector<uint8_t> data(chunk_size * dim);

  file.seekg(start_idx * 1 * dim+offset, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()),  chunk_size * dim);
  const double scaleParameter = 100;
  cout<<" rank  "<<rank<<"  data reading  completed"<<endl;

  for (INDEX_TYPE i = 0; i < chunk_size; ++i) {
    std::vector<float> vec(dim);
    std::copy(data.begin() + i * dim, data.begin() + (i + 1) * dim, vec.begin());
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [scaleParameter](float value) { return value * scaleParameter; });

    (*datamatrix)[i]=vec;
  }
}

static void  read_txt(string filename, ValueType2DVector<VALUE_TYPE>* datamatrix,
                     INDEX_TYPE no_of_datapoints,int dim, int rank, int world_size, INDEX_TYPE offset=8) {
  std::ifstream infile(filename); // Open the file for reading
  std::vector<std::vector<VALUE_TYPE>> data; // Vector to hold the loaded data

  if (infile) {
    std::string line;
    while (std::getline(infile, line)) {
      std::vector<VALUE_TYPE> row;
      std::istringstream iss(line);
      VALUE_TYPE value;

      // Read each value (label followed by features)
      while (iss >> value) {
        row.push_back(value);
      }

      // Add the row (data vector) to the data vector
      data.push_back(row);
    }
    datamatrix->resize(data.size());
    for(int i=0;i<data.size();i++){
      (*datamatrix)[i]=data[i];
    }

  } else {
    std::cerr << "Error opening file: " << filename << std::endl;
  }
}


static void  read_fbin_sparse(string filename, Eigen::SparseMatrix<float, Eigen::RowMajor> &output,
                      INDEX_TYPE no_of_datapoints,int dim, int rank, int world_size, INDEX_TYPE offset=8) {
  cout<<" rank  "<<rank<<"  openinig file "<<filename<<endl;
  std::ifstream file(filename, std::ios::binary);
  cout<<" rank  "<<rank<<"  openinig file "<<filename<<endl;
  if (!file.is_open()) {
    // Handle file opening error
    std::cerr << "Error: Unable to open the file " << filename << std::endl;
    return;
  }
  cout<<" rank  "<<rank<<"  openinig completed "<<filename<<endl;
  int nvecs=no_of_datapoints;

  //  file.read(reinterpret_cast<char*>(&nvecs), sizeof(int));
  //  file.read(reinterpret_cast<char*>(&dim), sizeof(int));
  //
  //  cout<<" rank  "<<rank<<"  nvecs "<<nvecs<<" dim "<<dim<<endl;

  INDEX_TYPE chunk_size = no_of_datapoints / world_size;
  INDEX_TYPE start_idx =rank*chunk_size;
  INDEX_TYPE end_index = 0;
  if (rank < world_size - 1){
    end_index = (rank+1) * chunk_size -1;
  }else if (rank == world_size - 1){
    end_index = std::min((rank+1) * chunk_size -1,no_of_datapoints-1);
    chunk_size = no_of_datapoints-(rank)*chunk_size;
  }

  if (chunk_size == -1) {
    chunk_size = no_of_datapoints - start_idx;
  }
//  datamatrix->resize(chunk_size, vector<VALUE_TYPE> (dim));

  std::vector<Eigen::Triplet<double>> triplets;


  cout<<" rank  "<<rank<<"  selected chunk size  "<<chunk_size<<" starting "<<start_idx<<endl;
  std::vector<float> data(chunk_size * dim);

  file.seekg(start_idx * 4 * dim+offset, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), sizeof(float) * chunk_size * dim);
  const double scaleParameter = 1;
  cout<<" rank  "<<rank<<"  data reading  completed"<<endl;

  for (INDEX_TYPE i = 0; i < chunk_size; ++i) {
    std::vector<float> vec(dim);
    std::copy(data.begin() + i * dim, data.begin() + (i + 1) * dim, vec.begin());
    for (int j = 0; j < dim; ++j) {
      float value = vec[j];
      if (value != 0) {
//        cout<<"( "<<i<<" "<<j<<" "<<value<<")"<<endl;
        triplets.push_back(Eigen::Triplet<double>(j,i,value*scaleParameter));
      }
    }
  }
  output.setFromTriplets(triplets.begin(), triplets.end());
  cout<<" rank  "<<rank<<"  data loading completed"<<endl;
}

static std::vector<std::pair<int, float>> processNonZeroEntries(const std::vector<float>& data, int dim, float scaleParameter) {
  std::vector<std::pair<int, float>> result;
  result.reserve(dim); // Reserve space for maximum possible non-zero entries

  for (int i = 0; i < dim; ++i) {
    float value = data[i];
    if (value != 0) {
      result.emplace_back(i, value * scaleParameter);
    }
  }

  return result;
}

static void read_fbin_with_MPI(string filename, ValueType2DVector<VALUE_TYPE>* datamatrix,
                               INDEX_TYPE no_of_datapoints, int dim, Process3DGrid* grid) {
  MPI_File file;
  MPI_Status status;

  int rank = grid->rank_in_col;
  int world_size = grid->col_world_size;

  cout << "Rank " << rank << " opening file " << filename << endl;

  MPI_File_open(grid->col_world, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

  int nvecs, global_dim;

  if (rank == 0) {
    MPI_File_read(file, &nvecs, 1, MPI_INT, &status);
    MPI_File_read(file, &global_dim, 1, MPI_INT, &status);
  }

//  MPI_Offset file_size;
//  MPI_File_get_size(file, &file_size);

  // Broadcast global_dim to all processes
  MPI_Bcast(&global_dim, 1, MPI_INT, 0, grid->col_world);

  cout<<" rank  "<<rank<<"  No of vectors  "<<nvecs<<" global_dim "<<global_dim<<endl;

  if (dim != global_dim) {
    cout << "Error: Dimension mismatch!" << endl;
    MPI_Abort(grid->col_world, 1);
    return;
  }

//  no_of_datapoints =100000;
  INDEX_TYPE chunk_size = no_of_datapoints / world_size;
  INDEX_TYPE start_idx = rank * chunk_size;
//  INDEX_TYPE start_idx = 0;
  INDEX_TYPE end_index = (rank < world_size - 1) ? ((rank + 1) * chunk_size - 1) : (no_of_datapoints - 1);
  chunk_size = end_index - start_idx + 1;
  cout<<" rank  "<<rank<<"  selected chunk size  "<<chunk_size<<" starting "<<start_idx<<" end index "<<end_index<<endl;
  datamatrix->resize(chunk_size, vector<VALUE_TYPE>(dim));

  //MPI_File_set_view(file, start_idx * 4 * dim + offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
  MPI_Offset file_offset = start_idx * 4 * dim + 8;
  INDEX_TYPE  data_offset =0;
  INDEX_TYPE max_read_chunk = 10000;
  INDEX_TYPE remaining = chunk_size;
  INDEX_TYPE reading_chunk = min(remaining,max_read_chunk);
  do{
    INDEX_TYPE  total_size = reading_chunk * dim;
     shared_ptr<vector<VALUE_TYPE>> data = make_shared<vector<VALUE_TYPE>>(total_size);
     if (rank==0) {
       cout << " rank  " << rank << "  data size  " << (*data).size()<<" data offset "<<data_offset << endl;
     }
      int total_bytes =   reading_chunk*4;
      MPI_File_read_at_all(file, file_offset, (*data).data(),total_bytes, MPI_CHAR, MPI_STATUS_IGNORE);

      const double scaleParameter = 10000;

      for (INDEX_TYPE i = 0; i < reading_chunk; ++i) {
        vector<VALUE_TYPE> vec(dim);
        copy( (*data).begin() + i * dim, (*data).begin() + (i + 1) * dim, vec.begin());
//        transform(vec.begin(), vec.end(), vec.begin(),
//                  [scaleParameter](double value) {
//                    return value * scaleParameter; });
        if (rank==0) {
          cout << " rank  " << rank << "  i  " <<i << endl;
          for (int k = 0; k < dim; k++) {
            cout << vec[k] << " ";
          }
          cout << endl;
        }
        uint64_t index = i + data_offset;
//        (*datamatrix)[index] = vec;
      }
      remaining = remaining - reading_chunk;
      file_offset = file_offset + total_bytes;
      data_offset = data_offset +reading_chunk;
      reading_chunk = min(remaining,max_read_chunk);
  }while(reading_chunk>0);

//  MPI_Status status_read;



//  int error_code;
//  MPI_Error_class(status_read.MPI_ERROR, &error_code);
//  if (error_code != MPI_SUCCESS) {
//    char error_string[MPI_MAX_ERROR_STRING];
//    int length;
//    MPI_Error_string(status_read.MPI_ERROR, error_string, &length);
//    cout << " MPI File Read Error: "<< rank << error_string << endl;
//    // Handle the error or terminate the program
//  }else {
//    cout << " rank  " << rank << " MPI file read success " << endl;
//  }

//  ofstream fout;
//  string file_name = "/pscratch/sd/i/isjarana/dist_viz_datasets/large_datasets/YANDEX/data_"+to_string(rank)+".txt";
//  fout.open(file_name, std::ios_base::app);
//   for(int i=0;i<chunk_size;i++){
//     for(int j=0;j<dim;j++){
//      fout<<(*datamatrix)[i][j]<<" ";
//     }
//     fout<<endl;
//   }

//  fout.close();
  MPI_File_close(&file);
}

static void parallel_read_MM(string file_path, vector<Tuple<VALUE_TYPE>> *coords,
                      bool copy_col_to_value) {
  MPI_Comm WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

  int proc_rank, num_procs;
  MPI_Comm_rank(WORLD, &proc_rank);
  MPI_Comm_size(WORLD, &num_procs);

  shared_ptr<CommGrid> simpleGrid;
  simpleGrid.reset(new CommGrid(WORLD, num_procs, 1));

  SpParMat<int64_t, float, SpDCCols<int64_t, float>> G(simpleGrid);

  INDEX_TYPE nnz;

  G.ParallelReadMM(file_path, true, maximum<float>());

  nnz = G.getnnz();
  if (proc_rank == 0) {
    cout << "File reader read " << nnz << " nonzeros." << endl;
  }
  SpTuples<int64_t, float> tups(G.seq());
  tuple<int64_t, int64_t, float> *values = tups.tuples;


  coords->resize(tups.getnnz());

#pragma omp parallel for
  for (int i = 0; i < tups.getnnz(); i++) {
    (*coords)[i].row = get<0>(values[i]);
    (*coords)[i].col = get<1>(values[i]);
    if (copy_col_to_value) {
      (*coords)[i].value = get<1>(values[i]);
    } else {
      (*coords)[i].value = get<2>(values[i]);
    }
  }

  int rowIncrement = G.getnrow() / num_procs;

#pragma omp parallel for
  for (int i = 0; i < (*coords).size(); i++) {
    (*coords)[i].row += rowIncrement * proc_rank;
  }
}

static void read_txt_dist(std::string filename, VALUE_TYPE* nCoordinates,
                          int no_of_datapoints, int dim, int rank, int world_size, INDEX_TYPE offset = 0) {
  std::cout << "Rank " << rank << " opening file " << filename << std::endl;
  std::ifstream file(filename);

  if (!file.is_open()) {
    // Handle file opening error
    std::cerr << "Error: Unable to open the file " << filename << std::endl;
    return;
  }
  std::cout << "Rank " << rank << " opened file " << filename << std::endl;

  INDEX_TYPE chunk_size = no_of_datapoints / world_size;
  INDEX_TYPE start_idx = rank * chunk_size;
  INDEX_TYPE end_index = 0;

  if (rank < world_size - 1) {
    end_index = (rank + 1) * chunk_size - 1;
  } else if (rank == world_size - 1) {
    end_index = std::min((rank + 1) * chunk_size - 1, no_of_datapoints - 1);
    chunk_size = no_of_datapoints - rank * chunk_size;
  }

  if (chunk_size == -1) {
    chunk_size = no_of_datapoints - start_idx;
  }

  std::cout << "Rank " << rank << " selected chunk size " << chunk_size << " starting " << start_idx << std::endl;

  // Skip lines up to start_idx
  std::string line;
  for (INDEX_TYPE i = 0; i < start_idx + offset && std::getline(file, line); ++i);

  // Read and parse the data
  for (INDEX_TYPE i = 0; i < chunk_size; ++i) {
    if (std::getline(file, line)) {
      std::istringstream ss(line);
      int nodeId;
      VALUE_TYPE value1, value2;

      ss >> nodeId >> value1 >> value2;  // Read nodeId and values

      // Store values in row-major order
      nCoordinates[i * dim] = value1;
      nCoordinates[i * dim + 1] = value2;
    }
  }

  std::cout << "Rank " << rank << " data reading completed" << std::endl;
}


static int reverse_int (int i){
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

};
}

