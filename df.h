//////////////////////////////////////////////////////////////////////////////
// Thymos Dataframe (c) 2022-6 by Zachary R. James (@thymos)
// The Dataframe struct and associated methods make core preprocessing possible
// st you can run ml models easily and quickly. The main goals are as follows:
//
// (i) importing csv files (though later formats may come)
// (ii) printing df, subsets, labels, & exporting to tensors
// (iii) pruning columns by label / indices
// (iv) splitting & sampling (by rows/cols, train-test/k-fold splits, &c)
// (v) encoding
// (vi) normalization / scaling (though should this go int the Tensor ops???)
// (vii) feature engineering
//
// (viii) time series-associated methods
// (ix) basic math stats
// (x) optimize this to leverate different retrieval locations
// (xi) optimized reading for large datasets
// (xii) pull data from apis / clouds (ex: yahoo finance, kaggle)
// (xiii) compare speed with other solutions
//////////////////////////////////////////////////////////////////////////////

// Novemeber 2025: working on the rewrite of these basec on the tests I am building (df_test.c)


#if ! defined(__THYMOS_DF_H__)
#define __THYMOS_DF_H__

// INCLUDES //
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "ml.h" // for the tensor struct export 
#include "la.h" // need to export to and maybe mirror some of these fns (noramlization/encoding for ex)

// forward-declaration

// DATAFRAME //

// used to automatically detect numbers (else sets everything to default string)
typedef enum
{
  TH_COLUMN_TYPE_STRING, // default
  TH_COLUMN_TYPE_FLOAT64 // value_t currently
} th_ColumnType;

typedef struct
{
    char *name;  // from first row of csv if named
    th_ColumnType type;
    size_t size;

    union
    {
        double *float64_data;
        char *string_data;
    } data;

} th_Column;

typedef struct
{
    th_Column *columns;
    size_t n_rows, n_columns;
} th_DataFrame;

th_Error th_init_dataframe(th_DataFrame *p, size_t n_rows, size_t n_columns);
th_Error th_destroy_dataframe(th_DataFrame *p);
th_Error th_print_dataframe_column_labels(th_DataFrame *p);
th_Error th_read_csv(const char *file_path, th_DataFrame *df, int MAX_LINE_LENGTH, bool data_has_top_label_row);
th_Error th_read_sql_lite(const char *file_path, th_DataFrame *df);
th_Error th_free_dataframe(th_DataFrame *p);
int th_compare_char_ptr(const void *a, const void *b);
th_Error th_remove_dataframe_columns_by_label(th_DataFrame *p, size_t num_cols, ...);

th_Error th_train_test_split_dataframe(th_DataFrame data, th_DataFrame *X_train, th_DataFrame *X_test, th_DataFrame *y_train, th_DataFrame *y_test, double test_size, int random_state);
th_Error th_train_test_split(th_DataFrame data_matrix, th_DataFrame *X_train_matrix, th_DataFrame *X_test_matrix, th_DataFrame *y_train_matrix, th_DataFrame *y_test_matrix, double test_size, int random_state);


#endif // __THYMOS_DF_H__
