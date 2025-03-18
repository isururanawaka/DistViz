#pragma once
#include "../net/process_3D_grid.hpp"
#include <map>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "../common/common.h"
#include "math_operations.hpp"
#include <memory>


using namespace hipgraph::distviz::knng;
using namespace hipgraph::distviz::common;
using namespace  hipgraph::distviz::net;

namespace hipgraph::distviz::knng {


template<typename INDEX_TYPE,typename VALUE_TYPE>
class GlobalTreeHandler {

private:
  int tree_depth;
  VALUE_TYPE *projected_matrix;
  VALUE_TYPE *projection_matrix;
  INDEX_TYPE local_dataset_size;
  int ntrees;
  INDEX_TYPE starting_data_index;
  INDEX_TYPE global_dataset_size;
  int data_dimension;
  Process3DGrid* grid;

  // multiple trees
  unique_ptr<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>> trees_data_ptr;
  unique_ptr<ValueType2DVector<VALUE_TYPE>> trees_splits_ptr;
  unique_ptr<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>> trees_leaf_first_indices_all_ptr;
  unique_ptr<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>> trees_leaf_first_indices_ptr;

  ValueType2DVector<VALUE_TYPE>* data_points_ptr;

  unique_ptr<ValueType2DVector<int>> index_to_tree_leaf_mapper_ptr;

  unique_ptr<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>> trees_leaf_first_indices_rearrange_ptr;

public:

  GlobalTreeHandler(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix,
                    Process3DGrid* grid, INDEX_TYPE local_dataset_size,
                    int dimension, int tree_depth, int ntrees,
                    INDEX_TYPE global_dataset_size) {
    this->tree_depth = tree_depth;
    this->local_dataset_size = local_dataset_size;
    this->projected_matrix = projected_matrix;
    this->projection_matrix = projection_matrix;
    this->global_dataset_size = global_dataset_size;
    this->data_dimension = dimension;

    this->ntrees = ntrees;

    this->trees_data_ptr = make_unique<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>>(ntrees);
    this->trees_splits_ptr = make_unique<ValueType2DVector<VALUE_TYPE>>(ntrees);
    this->trees_leaf_first_indices_ptr =  make_unique<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>>(ntrees);
    this->trees_leaf_first_indices_rearrange_ptr =  make_unique<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>>(ntrees);
    this->trees_leaf_first_indices_all_ptr =  make_unique<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>>(ntrees);

    this->starting_data_index = (global_dataset_size / grid->col_world_size) * grid->rank_in_col;

    this->grid = grid;
  }

  int select_next_candidate (vector<vector<vector<vector<LeafPriority >>>> &candidate_mapping,
                            vector<vector<int>> &final_tree_leaf_mapping, int current_tree,
                            int selecting_tree, int selecting_leaf, int previouse_leaf,
                            int total_leaf_size) {

    vector <LeafPriority> vec = candidate_mapping[current_tree][previouse_leaf][selecting_tree];
    sort(vec.begin(), vec.end(),
         [](const LeafPriority &lhs, const LeafPriority &rhs) {
           if (lhs.priority > rhs.priority) {
             return true;
           }else {
             return false;
           }
         });

    for ( int i = 0; i<vec.size (); i++) {
      LeafPriority can_leaf = vec[i];
      int id = can_leaf.leaf_index;
      bool candidate = true;

      // checking already taken
      for ( int j = selecting_leaf - 1; j >= 0; j--) {
        if (final_tree_leaf_mapping[j][selecting_tree] == id) {
          candidate = false;
          break;
        }
      }

      if (!candidate) {
        continue;
      }

      for ( int j = 0; j<total_leaf_size; j++) {
        vector <LeafPriority> neighbour_vec = candidate_mapping[current_tree][j][selecting_tree];
        if (j != previouse_leaf and neighbour_vec[0].priority > can_leaf.priority and neighbour_vec[0] .leaf_index == can_leaf.leaf_index) {
          candidate = false;
          break;
        }
      }

      if (candidate) {

        final_tree_leaf_mapping[selecting_leaf][selecting_tree] = can_leaf.leaf_index;
        return can_leaf.leaf_index;
      }

    }
    return -1;
  }

  void grow_global_tree(ValueType2DVector<VALUE_TYPE>* data_points) {

    if (tree_depth <= 0 || tree_depth > log2(local_dataset_size))
    {
      throw std::out_of_range (" depth should be in range [1,....,log2(rows)]");
    }

    if (ntrees <= 0)
    {
      throw std::out_of_range (" no of trees should be greater than zero");
    }

    data_points_ptr = data_points; // convert this to pointer reference

    int total_split_size = 1 << (tree_depth + 1);
    int total_child_size = (1 << (tree_depth)) - (1 << (tree_depth - 1));


    index_to_tree_leaf_mapper_ptr = make_unique<ValueType2DVector<int>>(local_dataset_size);

    for (int k = 0; k < this->ntrees; k++)
    {
      (*trees_splits_ptr)[k] = vector<VALUE_TYPE> (total_split_size);
      (*trees_data_ptr)[k] = vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE>>>(this->tree_depth);
      (*trees_leaf_first_indices_ptr)[k] = vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE>>> (total_child_size);
      (*trees_leaf_first_indices_all_ptr)[k] = vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE>>> (total_child_size);
      (*trees_leaf_first_indices_rearrange_ptr)[k] = vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE> >> (total_child_size);

#pragma  omp parallel for
      for (int i = 0; i < this->tree_depth; i++)
      {
        (*trees_data_ptr)[k][i] = vector<DataNode<INDEX_TYPE,VALUE_TYPE>> (this->local_dataset_size);
      }
    }

    // storing projected data
#pragma  omp parallel for
    for (INDEX_TYPE j = 0; j < local_dataset_size; j++)
    {
      (*index_to_tree_leaf_mapper_ptr)[j] = vector<int> (ntrees);
      for (int k = 0; k < ntrees; k++)
      {
        for (int i = 0; i < tree_depth; i++)
        {
          auto index =tree_depth * k + i + j * tree_depth * ntrees;
          DataNode<INDEX_TYPE,VALUE_TYPE> dataPoint;

          dataPoint.value = projected_matrix[index];
          dataPoint.index = j + starting_data_index;
          (*trees_data_ptr)[k][i][j] = dataPoint;
        }
      }
    }


    for (int k = 0; k < ntrees; k++)
    {
      vector <vector<DataNode<INDEX_TYPE,VALUE_TYPE>>> child_data_tracker (total_split_size);
      vector<int> global_size_vector (total_split_size);
      child_data_tracker[0] = (*trees_data_ptr)[k][0];
      global_size_vector[0] = global_dataset_size;

      for (int i = 0; i < tree_depth - 1; i++)
      {
        grow_global_subtree (child_data_tracker, global_size_vector, i, k);

      }
    }
  }

  void grow_global_subtree(vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE>>> &child_data_tracker,
                           vector<int> &global_size_vector, int depth, int tree) {
    int current_nodes = (1 << (depth));
    int split_starting_index = (1 << (depth)) - 1;
    int next_split = (1 << (depth + 1)) - 1;

    if (depth == 0) {
      split_starting_index = 0;
    }

    //object for math operations
    unique_ptr<MathOp<INDEX_TYPE,VALUE_TYPE>> mathOp_ptr;

    vector<VALUE_TYPE> data(local_dataset_size);

    int total_data_count_prev = 0;
    vector<int> local_data_row_count (current_nodes);
    vector<int> global_data_row_count (current_nodes);

    int minimum_vector_size=INT32_MAX;
    for (int i = 0; i < current_nodes; i++) {
      vector <DataNode<INDEX_TYPE,VALUE_TYPE>> data_vector = child_data_tracker[split_starting_index + i];
      local_data_row_count[i] = data_vector.size ();
      //keep track of minimum child size
      if(minimum_vector_size>data_vector.size ()){
        minimum_vector_size = data_vector.size ();
      }
      global_data_row_count[i] = global_size_vector[split_starting_index + i];

#pragma omp parallel for
      for (int j = 0; j < data_vector.size (); j++)
      {
        data[j + total_data_count_prev] = data_vector[j].value;
      }
      //        cout << " rank " << rank << " level " << depth << " child" << i << " preprocess completed " << endl;
      total_data_count_prev += data_vector.size ();
    }

    // calculation of bins
//        int no_of_bins = 1 + (3.322 * log2(minimum_vector_size));
    int no_of_bins = 7;

    //calculation of distributed median
    VALUE_TYPE *result = mathOp_ptr.get()->distributed_median (data, local_data_row_count, current_nodes,
                                                   global_data_row_count,
                                                   no_of_bins,
                                                   StorageFormat::RAW, grid->rank_in_col);

    for (int i = 0; i < current_nodes; i++)
    {
      int left_index = (next_split + 2 * i);
      int right_index = left_index + 1;

      int selected_leaf_left = left_index - (1 << (tree_depth - 1)) + 1;
      int selected_leaf_right = selected_leaf_left + 1;

      vector<DataNode<INDEX_TYPE,VALUE_TYPE>> data_vector = child_data_tracker[split_starting_index + i];
//      vector<VALUE_TYPE> data(data_vector.size());

      auto median =  (result)[i];

      if (grid->rank_in_col==0){
        cout<<" rank "<<grid->rank_in_col<<" median for depth "<<depth<<" median "<< median<<endl;
      }

      //store median in tree_splits
      (*trees_splits_ptr)[tree][split_starting_index + i] = median;

      vector<DataNode<INDEX_TYPE,VALUE_TYPE>> left_childs_global;
      vector<DataNode<INDEX_TYPE,VALUE_TYPE>> right_childs_global;

#pragma omp parallel
      {
        vector <DataNode<INDEX_TYPE,VALUE_TYPE>> left_childs;
        vector <DataNode<INDEX_TYPE,VALUE_TYPE>> right_childs;
#pragma omp for  nowait
        for (INDEX_TYPE k = 0; k < data_vector.size (); k++)
        {
          auto index = data_vector[k].index;

          auto selected_index = index - starting_data_index;
          DataNode<INDEX_TYPE,VALUE_TYPE> selected_data = (*trees_data_ptr)[tree][depth + 1][selected_index];

          if (data_vector[k].value <= median)
          {
            left_childs.push_back (selected_data);
            if (depth == this->tree_depth - 2)
            {
              //keep track of respective leaf of selected_index
              (*index_to_tree_leaf_mapper_ptr)[selected_index][tree] = selected_leaf_left;
            }
          }
          else
          {
            right_childs.push_back (selected_data);
            if (depth == this->tree_depth - 2)
            {
              //keep track of respective leaf of selected_index
              (*index_to_tree_leaf_mapper_ptr)[selected_index][tree] = selected_leaf_right;
            }
          }
        }

#pragma omp critical
        {
          left_childs_global.insert (left_childs_global.end (), left_childs.begin (), left_childs.end ());
          right_childs_global.insert (right_childs_global.end (), right_childs.begin (), right_childs.end ());
        }
      }

//      cout<<" rank "<<grid->rank_in_col<< " left child size" <<left_childs_global.size()<<" right child size "<<right_childs_global.size()<<" depth"<<depth<<endl;
      child_data_tracker[left_index] = left_childs_global;
      child_data_tracker[right_index] = right_childs_global;
      if (depth == tree_depth - 2) {
//        cout<<" rank "<<grid->rank_in_col<< " left child size" <<left_childs_global.size()<<" right child size "<<right_childs_global.size()<<endl;
        (*trees_leaf_first_indices_ptr)[tree][selected_leaf_left] = left_childs_global;
        (*trees_leaf_first_indices_ptr)[tree][selected_leaf_right] = right_childs_global;
      }
    }

    if (depth == this->tree_depth - 2) {
      return;
    }

    this->derive_global_datavector_sizes(child_data_tracker,global_size_vector,current_nodes,next_split);

//    free (result);
  }

  void calculate_tree_leaf_correlation() {
    vector<vector<vector<vector<LeafPriority>>>> candidate_mapping =
        vector<vector<vector<vector<LeafPriority>>>> (ntrees);

    int total_leaf_size = (1 << (tree_depth)) - (1 << (tree_depth - 1));

    vector <vector<int>> final_tree_leaf_mapping (total_leaf_size);

    int total_sending = ntrees * total_leaf_size * ntrees;
    int *my_sending_leafs = new int[total_sending] ();

    int total_receiving = ntrees * total_leaf_size * ntrees * grid->col_world_size;

    int *total_receiving_leafs = new int[total_receiving] ();

    int *send_count = new int[grid->col_world_size] ();
    int *disps_send = new int[grid->col_world_size] ();
    int *recieve_count = new int[grid->col_world_size] ();
    int *disps_recieve = new int[grid->col_world_size] ();

    for (int i = 0; i < grid->col_world_size; i++)
    {
      send_count[i] = total_sending;
      disps_send[i] = 0;
      recieve_count[i] = total_sending;
      disps_recieve[i] = (i > 0) ? (disps_recieve[i - 1] + recieve_count[i - 1]) : 0;
    }

    vector<vector<vector<vector<float>>>> correlation_matrix =
        vector<vector<vector<vector<float>>>> (ntrees);

    for (int tree = 0; tree < ntrees; tree++)
    {
      correlation_matrix[tree] = vector < vector < vector < float>>>(total_leaf_size);
      candidate_mapping[tree] = vector < vector < vector < LeafPriority>>>(total_leaf_size);
      for (int leaf = 0; leaf < total_leaf_size; leaf++)
      {
        correlation_matrix[tree][leaf] = vector < vector < float >> (ntrees);
        candidate_mapping[tree][leaf] = vector < vector < LeafPriority >> (ntrees);
        vector <DataNode<INDEX_TYPE,VALUE_TYPE>> data_points = (*trees_leaf_first_indices_ptr)[tree][leaf];
        for (int j = 0; j < ntrees; j++)
        {
          correlation_matrix[tree][leaf][j] = vector<float> (total_leaf_size, 0);
        }
#pragma omp parallel for
        for (int c = 0; c < data_points.size (); c++)
        {
          int selec_index = data_points[c].index - starting_data_index;
          vector<int> vec = (*index_to_tree_leaf_mapper_ptr)[selec_index];
          for (int j = 0; j < vec.size (); j++)
          {
#pragma omp atomic
            correlation_matrix[tree][leaf][j][vec[j]] += 1;
          }
        }
      }
    }

    for (int tree = 0; tree < ntrees; tree++)
    {
#pragma omp parallel for
      for (int leaf = 0; leaf < total_leaf_size; leaf++)
      {
        for (int c = 0; c < ntrees; c++)
        {
          vector <DataNode<INDEX_TYPE,VALUE_TYPE>> data_points = (*trees_leaf_first_indices_ptr)[tree][leaf];
          vector<float> result = 	vector<float> (total_leaf_size, 0);
          int size = data_points.size ();
          std::transform (correlation_matrix[tree][leaf][c].begin (), correlation_matrix[tree][leaf][c].end (),
                         result.begin(), [&] (float x)
                         { return (x / size) * 100; });
          correlation_matrix[tree][leaf][c] = result;
          int selected_leaf = std::max_element (correlation_matrix[tree][leaf][c].begin (),
                                               correlation_matrix[tree][leaf][c].end ()) -
                              correlation_matrix[tree][leaf][c].begin ();
          int count = c + leaf * ntrees + tree * total_leaf_size * ntrees;
          my_sending_leafs[count] = selected_leaf;
        }
      }
    }
    auto t = start_clock();
    MPI_Allgatherv (my_sending_leafs, total_sending, MPI_INT, total_receiving_leafs,
                   recieve_count, disps_recieve, MPI_INT, MPI_COMM_WORLD);
    stop_clock_and_add(t, "KNNG Communication Time");

    for (int j = 0; j < ntrees; j++)
    {
      for (int k = 0; k < total_leaf_size; k++)
      {
        final_tree_leaf_mapping[k] = vector<int> (ntrees, -1);
        for (int m = 0; m < ntrees; m++)
        {
          candidate_mapping[j][k][m] = vector<LeafPriority> (total_leaf_size);
        }
      }
    }

#pragma omp parallel for
    for (int l = 0; l < ntrees * total_leaf_size * ntrees; l++)
    {

      int m = l % ntrees;
      int temp_val = (l - m) / ntrees;
      int j = temp_val / total_leaf_size;
      int k = temp_val % total_leaf_size;

      for (int n = 0; n < total_leaf_size; n++) {
        LeafPriority priorityMap;
        priorityMap.priority = 0;
        priorityMap.leaf_index = n;
        candidate_mapping[j][k][m][n]=priorityMap;
      }

      vector<int> vec;
      for (int p = 0; p < grid->col_world_size; p++)
      {
        int id = p * total_sending + j * total_leaf_size * ntrees + k * ntrees + m;
        int value = total_receiving_leafs[id];
        vec.push_back (value);
      }
      sortByFreq (vec, candidate_mapping[j][k][m], grid->col_world_size);
    }


#pragma  omp parallel for
    for (int k = 0; k < total_leaf_size; k++)
    {
      int prev_leaf = k;
      for (int m = 0; m < ntrees; m++)
      {
        int current_tree = m == 0 ? 0 : m - 1;
        prev_leaf = select_next_candidate (candidate_mapping, final_tree_leaf_mapping, current_tree, m, k, prev_leaf,
                                          total_leaf_size);
      }
    }

    for (int i = 0; i < ntrees; i++)
    {
#pragma  omp parallel for
      for (int k = 0; k < total_leaf_size; k++)
      {
        int leaf_index = final_tree_leaf_mapping[k][i];
        //clustered data is stored in the rearranged tree leaves.
        (*trees_leaf_first_indices_rearrange_ptr)[i][k] = (*trees_leaf_first_indices_ptr)[i][leaf_index];
      }
    }

    delete[]
        my_sending_leafs;
    delete[]
        total_receiving_leafs;
    delete[]
        send_count;
    delete[]
        disps_send;
    delete[]
        recieve_count;
    delete[]
        disps_recieve;
  }

  void derive_global_datavector_sizes(vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE>>> &child_data_tracker,
                                      vector<int> &global_size_vector,int current_nodes, int next_split) {
    // Displacements in the receive buffer for MPI_GATHERV
    int *disps = new int[grid->col_world_size];

    // Displacement for the first chunk of data - 0
    for (int i = 0; i < grid->col_world_size; i++)
    {
      disps[i] = (i > 0) ? (disps[i - 1] + 2 * current_nodes) : 0;
    }

    int *total_counts = new int[2 * grid->col_world_size * current_nodes]();

    int *process_counts = new int[grid->col_world_size]();

    for (int k = 0; k < grid->col_world_size; k++)
    {
      process_counts[k] = 2 * current_nodes;
    }
    for (int j = 0; j < current_nodes; j++)
    {
      int id = (next_split + 2 * j);
      total_counts[2 * j + grid->rank_in_col * current_nodes * 2] = child_data_tracker[id].size ();
      total_counts[2 * j + 1 + grid->rank_in_col * current_nodes * 2] = child_data_tracker[id + 1].size ();
    }

    auto t = start_clock();
    MPI_Allgatherv (MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);
    stop_clock_and_add(t, "KNNG Communication Time");

    for (int j = 0; j < current_nodes; j++)
    {
      int left_totol = 0;
      int right_total = 0;
      int id = (next_split + 2 * j);
      for (int k = 0; k < grid->col_world_size; k++)
      {

        left_totol = left_totol + total_counts[2 * j + k * current_nodes * 2];
        right_total = right_total + total_counts[2 * j + 1 + k * current_nodes * 2];
      }

      global_size_vector[id] = left_totol;
      global_size_vector[id + 1] = right_total;
    }

    free (process_counts);
    free (total_counts);
    free (disps);
  }


  void collect_similar_data_points_of_all_trees(vector<VALUE_TYPE>* receive_values_ptr,bool use_data_locality_optimization,
                                   vector<set<INDEX_TYPE>>* process_to_index_set_ptr,
                                                           map<INDEX_TYPE,INDEX_TYPE>* local_to_global_map,
                                                           map<INDEX_TYPE,vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>>& local_nn_map, int nn) {


    int total_leaf_size = (1 << (tree_depth)) - (1 << (tree_depth - 1));
    int leafs_per_node = total_leaf_size / grid->col_world_size;
    cout<<"total_leaf_size"<<total_leaf_size<<" leafs_per_node "<<leafs_per_node<<endl;

    for (int tree = 0; tree < ntrees; tree++) {
      int process = 0;
      for (int i = 0; i < total_leaf_size; i++) {
        if (i > 0 && i % leafs_per_node == 0) {
          process++;
        }
        vector<DataNode<INDEX_TYPE, VALUE_TYPE>> all_points =
            (use_data_locality_optimization)? (*trees_leaf_first_indices_rearrange_ptr)[tree][i]:(*trees_leaf_first_indices_ptr)[tree][i];

        std::set<INDEX_TYPE>& firstSet = (*process_to_index_set_ptr)[process];
        for (INDEX_TYPE j=0;j<all_points.size();j++){
          firstSet.insert(all_points[j].index);
        }
      }
    }


    shared_ptr<vector<int>> send_indices_count_ptr =  make_shared<vector<int>>(grid->col_world_size);
    shared_ptr<vector<int>> send_disps_indices_count_ptr =  make_shared<vector<int>>(grid->col_world_size);
    shared_ptr<vector<int>> send_values_count_ptr =  make_shared<vector<int>>(grid->col_world_size);
//    shared_ptr<vector<int>> send_disps_values_count_ptr =  make_shared<vector<int>>(grid->col_world_size);

    shared_ptr<vector<MPI_Aint>> send_disps_values_count_ptr =  make_shared<vector<MPI_Aint>>(grid->col_world_size);

    shared_ptr<vector<int>> receive_indices_count_ptr =  make_shared<vector<int>>(grid->col_world_size);
    shared_ptr<vector<int>> receive_disps_indices_count_ptr =  make_shared<vector<int>>(grid->col_world_size);
    shared_ptr<vector<int>> receive_values_count_ptr =  make_shared<vector<int>>(grid->col_world_size);
//    shared_ptr<vector<int>> receive_disps_values_count_ptr =  make_shared<vector<int>>(grid->col_world_size);
    shared_ptr<vector<MPI_Aint>> receive_disps_values_count_ptr =  make_shared<vector<MPI_Aint>>(grid->col_world_size);

    MPI_Datatype* data_type = new MPI_Datatype[grid->col_world_size];

    auto total_send_count=0;
    for(int i=0;i<grid->col_world_size;i++){
//      if (i!= grid->rank_in_col){
        (*send_indices_count_ptr)[i]= (*process_to_index_set_ptr)[i].size();
//      }else {
//        (*send_indices_count_ptr)[i] = 0;
//      }
      (*send_values_count_ptr)[i]= (*send_indices_count_ptr)[i]*data_dimension;
      total_send_count +=(*send_indices_count_ptr)[i];
      (*send_disps_indices_count_ptr)[i]=(i>0)?(*send_disps_indices_count_ptr)[i-1]+(*send_indices_count_ptr)[i-1]:0;

      MPI_Aint displacement = (i > 0) ? (*send_disps_values_count_ptr)[i - 1] + (*send_indices_count_ptr)[i - 1] * data_dimension : 0;
      (*send_disps_values_count_ptr)[i] = displacement;
//      (*send_disps_values_count_ptr)[i]=(i>0)?(*send_disps_values_count_ptr)[i-1]+(*send_indices_count_ptr)[i-1]*data_dimension:0;

      data_type[i]=MPI_VALUE_TYPE;
//      cout<<" rank "<<grid->rank_in_col<<" sending values "<<(*send_values_count_ptr)[i]<<" to rank "<<i<<endl;
    }

    auto t = start_clock();
//    //send indices count
    MPI_Alltoall ((*send_indices_count_ptr).data(),1 , MPI_INT,(*receive_indices_count_ptr).data(), 1,MPI_INT, grid->col_world);

    cout<<" first data point collection passed rank "<<grid->rank_in_col <<endl;

    stop_clock_and_add(t, "KNNG Communication Time");


    auto total_receive_count=0;
    for(int i=0;i<grid->col_world_size;i++) {
      (*receive_disps_indices_count_ptr)[i]=(i>0)?(*receive_disps_indices_count_ptr)[i-1]+(*receive_indices_count_ptr)[i-1]:0;
      total_receive_count += (*receive_indices_count_ptr)[i];
//      (*receive_disps_values_count_ptr)[i]=(i>0)?(*receive_disps_values_count_ptr)[i-1]+(*receive_indices_count_ptr)[i-1]*data_dimension:0;

      MPI_Aint displacement = (i>0)?(*receive_disps_values_count_ptr)[i-1]+(*receive_indices_count_ptr)[i-1]*data_dimension:0;
      (*receive_disps_values_count_ptr)[i]=displacement;

      (*receive_values_count_ptr)[i]=(*receive_indices_count_ptr)[i]*data_dimension;
//      cout<<" rank "<<grid->rank_in_col<<" receiving count "<<(*receive_values_count_ptr)[i]<<" from rank "<<i<<endl;

    }
//
//
    uint64_t  total_send = static_cast<uint64_t>(total_send_count)*static_cast<uint64_t >(data_dimension);
    shared_ptr<vector<INDEX_TYPE>> send_indices_ptr =  make_shared<vector<INDEX_TYPE>>(total_send_count);
    shared_ptr<vector<VALUE_TYPE>> send_values_ptr =  make_shared<vector<VALUE_TYPE>>(total_send);

    uint64_t  total_receive= static_cast<uint64_t>(total_receive_count)*static_cast<uint64_t >(data_dimension);
    shared_ptr<vector<INDEX_TYPE>> receive_indices_ptr =  make_shared<vector<INDEX_TYPE>>(total_receive_count);
//    shared_ptr<vector<VALUE_TYPE>> receive_values_ptr =  make_shared<vector<VALUE_TYPE>>(total_receive);
    receive_values_ptr->resize(total_receive);
//
//    cout<<" MPI value initialization  passed rank "<<grid->rank_in_col <<endl;
//
    for(int i=0;i<grid->col_world_size;i++){
//      if (i != grid->rank_in_col) {
        auto offset = (*send_disps_indices_count_ptr)[i];

        std::set<INDEX_TYPE>& data_set = (*process_to_index_set_ptr)[i];
        auto it = data_set.begin();
        for (auto j = 0; j < (*send_indices_count_ptr)[i]; j++) {
          // Access the value using the iterator
          INDEX_TYPE in = offset + j;
          (*send_indices_ptr)[in] = (*it);
          ++it;
          auto index_trying = (*send_indices_ptr)[in] - starting_data_index;
          auto access_index_dim = (in) * data_dimension;
          for (int k = 0; k < data_dimension; ++k) {
            auto access_index_dim_d = access_index_dim + k;
             (*send_values_ptr)[access_index_dim_d] =(*data_points_ptr)[index_trying][k];
          }
//        }
      }
    }
//
    t = start_clock();


    MPI_Alltoallv((*send_indices_ptr).data(),(*send_indices_count_ptr).data(),(*send_disps_indices_count_ptr).data() , MPI_INDEX_TYPE,
                  (*receive_indices_ptr).data(), (*receive_indices_count_ptr).data(),
                  (*receive_disps_indices_count_ptr).data(),MPI_INDEX_TYPE, grid->col_world);

    cout<<" MPI value indices completed "<<grid->rank_in_col <<endl;

//    MPI_Alltoallv ((*send_values_ptr).data(),(*send_values_count_ptr).data(),
//                  (*send_disps_values_count_ptr).data() , MPI_VALUE_TYPE,(*receive_values_ptr).data(),
//                  (*receive_values_count_ptr).data(),(*receive_disps_values_count_ptr).data(),MPI_VALUE_TYPE, grid->col_world);


    vector<MPI_Request> send_requests(grid->col_world_size, MPI_REQUEST_NULL);
    vector<MPI_Request> recv_requests(grid->col_world_size, MPI_REQUEST_NULL);
    uint64_t send_offset =0;
    uint64_t receive_offset =0;
    int TAG_MULTIPLIER = 10000;
    cout<<" MPI send all  initiated "<<grid->rank_in_col <<endl;
    for(int i=0;i<grid->col_world_size;i++){
//      if (i!= grid->rank_in_col){
        VALUE_TYPE* send_buf = (*send_values_ptr).data() +send_offset;
        VALUE_TYPE* receive_buf = (*receive_values_ptr).data() +receive_offset;
        MPI_Isend(send_buf, (*send_values_count_ptr)[i], MPI_VALUE_TYPE, i, TAG_MULTIPLIER, grid->col_world, &send_requests[i]);
        MPI_Irecv(receive_buf, (*receive_values_count_ptr)[i], MPI_VALUE_TYPE, i, TAG_MULTIPLIER, grid->col_world, &recv_requests[i]);
        send_offset +=(*send_values_count_ptr)[i];
        receive_offset+=(*receive_values_count_ptr)[i];
//      }
    }
    cout<<" MPI send all  completed "<<grid->rank_in_col <<endl;
    MPI_Waitall(grid->col_world_size, send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(grid->col_world_size, recv_requests.data(), MPI_STATUSES_IGNORE);
    cout<<" MPI waiting  completed "<<grid->rank_in_col <<endl;


     cout<<" MPI Neighbour all to all completed "<<grid->rank_in_col <<endl;

     MPI_Barrier(grid->col_world);
//    cout<<" MPI value seinding passed rank "<<grid->rank_in_col <<endl;
     stop_clock_and_add(t, "KNNG Communication Time");

    for(auto i=0;i<total_receive_count;i++) {
      INDEX_TYPE receive_index = (*receive_indices_ptr)[i];
      (*local_to_global_map)[i]= receive_index;
      (local_nn_map)[receive_index] = vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>(nn);
    }

  }
};
}

