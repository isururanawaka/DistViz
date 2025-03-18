/**
 * This implements the CSR data structure. We use MKL routines to create CSR
 * from COO.
 */
#pragma once

#include "../common/common.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mpi.h>
#include <numeric>
#include <parallel/algorithm>
#include <string.h>
#include <set>

using namespace std;
using namespace hipgraph::distviz::common;

namespace hipgraph::distviz::embedding {

    template<typename INDEX_TYPE, typename VALUE_TYPE>
    class CSRLocal {

    public:
        INDEX_TYPE rows, cols;

        INDEX_TYPE max_nnz, num_coords = 0;

        bool transpose;

        unique_ptr <CSRHandle<INDEX_TYPE, VALUE_TYPE>> handler = nullptr;

        CSRLocal() {}

        CSRLocal(INDEX_TYPE rows, INDEX_TYPE cols, INDEX_TYPE max_nnz, Tuple<VALUE_TYPE> *coords, int num_coords,
                 bool transpose) {
            if (num_coords > 0) {

                this->transpose = transpose;
                this->num_coords = num_coords;
                this->rows = rows;
                this->cols = cols;
                this->handler = unique_ptr<CSRHandle<INDEX_TYPE, VALUE_TYPE>>(
                        new CSRHandle<INDEX_TYPE, VALUE_TYPE>());

                (handler.get())->values.resize(max_nnz == 0 ? 1 : max_nnz);
                (handler.get())->col_idx.resize(max_nnz == 0 ? 1 : max_nnz);
                (handler.get())->rowStart.resize((transpose) ? this->cols + 1 : this->rows + 1);


                if (!transpose) {
                    int expected_col = 0;
                    int current_row_value = 0;
                    set <INDEX_TYPE> dups;
                    int index = 0;
                    for (INDEX_TYPE i = 0; i < num_coords; i++) {
                        if (dups.insert(coords[i].row).second) {
                            if (expected_col == coords[i].row) {
                                (handler.get())->rowStart[index] = current_row_value;
                                index++;
                                expected_col++;
                            } else if (expected_col < coords[i].row) {
                                while (expected_col <= coords[i].row) {
                                    (handler.get())->rowStart[index] = current_row_value;
                                    index++;
                                    expected_col++;
                                }
                            }
                        }
                        (handler.get())->col_idx[i] = coords[i].col;
                        (handler.get())->values[i] = coords[i].value;
                        current_row_value++;
                    }

                    if (expected_col <= this->rows) {
                        while (expected_col <= this->rows) {
                            (handler.get())->rowStart[index] = current_row_value;
                            index++;
                            expected_col++;
                        }
                    }
                } else {
                    cout<<" executing non transpose csr conversion "<<endl;
                    int expected_col = 0;
                    int current_row_value = 0;
                    set <INDEX_TYPE> dups;
                    int index = 0;
                    for (INDEX_TYPE i = 0; i < num_coords; i++) {
                        if (dups.insert(coords[i].col).second) {
                            if (expected_col == coords[i].col) {
                                if (index >= handler.get()->rowStart.size()){
                                    cout<<" index and rowStart mismatch "<<coords[i].col<<" "<< index<<" "<<handler.get()->rowStart.size() <<endl;
                                }
                                (handler.get())->rowStart[index] = current_row_value;
                                index++;
                                expected_col++;
                            } else if (expected_col < coords[i].col) {
                                while (expected_col <= coords[i].col) {
                                    if (index >= handler.get()->rowStart.size()){
                                        cout<<" index and rowStart mismatch "<<coords[i].col<<" "<< index<<" "<<handler.get()->rowStart.size() <<endl;
                                    }
                                    (handler.get())->rowStart[index] = current_row_value;
                                    index++;
                                    expected_col++;
                                }
                            }
                        }
                        (handler.get())->col_idx[i] = coords[i].row;
                        (handler.get())->values[i] = coords[i].value;
                        current_row_value++;
                    }
                    if (expected_col <= this->cols) {
                        while (expected_col <= this->cols) {
//                            if (index >= handler.get()->rowStart.size()){
//                                cout<<" index and rowStart mismatch "<<coords[i].col<<" "<< index<<" "<<handler.get()->rowStart.size() <<endl;
//                            }
                            (handler.get())->rowStart[index] = current_row_value;
                            index++;
                            expected_col++;
                        }
                    }
                }
            }
        }

        ~CSRLocal() {
            //    mkl_sparse_destroy((handler.get())->mkl_handle);
        }
    };

} // namespace hipgraph::distviz::embedding
