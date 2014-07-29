package ALS

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"

	. "github.com/skelterjohn/go.matrix"
)

var (
	NA = math.NaN()
)

func errcheck(err error) {
	if err != nil {
		fmt.Printf("Error occured: %v", err)
	}
}

// Create the W matrix for the ALS algorithm..
func MakeWeightMatrix(mat *DenseMatrix) *DenseMatrix {
	values := mat.Array()
	newvalues := make([]float64, len(values))
	for i := 0; i < len(values); i++ {
		if values[i] == 0.0 || math.IsNaN(values[i]) {
			newvalues[i] = 0
		} else {
			newvalues[i] = 1
		}
	}
	return MakeDenseMatrix(newvalues, mat.Rows(), mat.Cols())
}

func MakeRatingMatrix(ratings []float64, rows, cols int) *DenseMatrix {
	return MakeDenseMatrix(ratings, rows, cols)
}

// creates the confidence matrix for the implicit ALS algorithm
func MakeCMatrix(mat *DenseMatrix) *DenseMatrix {
	values := mat.Array()
	newvalues := make([]float64, len(values))
	for i := 0; i < len(values); i++ {
		if values[i] == 0.0 || math.IsNaN(values[i]) {
			newvalues[i] = 1
		} else {
			newvalues[i] = 1 + 40*values[i] // the value of 20 for confidence was
		} // recommended by the aforementioned paper regarding implicit ALS
	}
	return MakeDenseMatrix(newvalues, mat.Rows(), mat.Cols())
}

// create X and Y matrices for the ALS algorithm
func MakeXY(mat *DenseMatrix, n_factors int, max_rating float64, seed int) (X, Y *DenseMatrix) {
	rand.Seed(int64(seed))
	rows := mat.Rows()
	cols := mat.Cols()
	X_data := make([]float64, rows*n_factors)
	Y_data := make([]float64, cols*n_factors)
	for i := 0; i < len(X_data); i++ {
		X_data[i] = max_rating * rand.Float64()
	}
	for j := 0; j < len(Y_data); j++ {
		Y_data[j] = max_rating * rand.Float64()
	}
	X = MakeDenseMatrix(X_data, rows, n_factors)
	Y = MakeDenseMatrix(Y_data, n_factors, cols)
	return
}

// adds up all the elements of the array
func sumMatrix(mat *DenseMatrix) (sum float64) {
	values := mat.Array()
	sum = float64(0)
	for i := 0; i < len(values); i++ {
		sum += values[i]
	}
	return
}

// Auxilliary function for Matrix Solver.
func swapCols(mat *DenseMatrix, i, j int) *DenseMatrix {
	p := mat.Copy()
	trans := p.Transpose()
	trans.SwapRows(i, j)
	toret := trans.Transpose()
	return toret
}

// Scales matrix mat by weight.
func simpleTimes(mat, weight *DenseMatrix) *DenseMatrix {
	if len(mat.Array()) != len(weight.Array()) {
		return nil
	}
	weightArray := weight.Array()
	matValues := mat.Array()
	for i := 0; i < len(matValues); i++ {
		matValues[i] = weightArray[i] * matValues[i]
	}
	return MakeDenseMatrix(matValues, mat.Rows(), mat.Cols())
}

// Gets the error for the alternating least squares algorithm. Used for Explicit ALS
func getErrorInline(W, q, X, Y *DenseMatrix) float64 {
	Q := q.Copy()
	dot, err := X.TimesDense(Y)
	errcheck(err)
	err = Q.SubtractDense(dot)
	errcheck(err)
	Prod := simpleTimes(Q, W)
	tosum := simpleTimes(Prod, Prod)
	sum := sumMatrix(tosum)
	return sum
}

// a function to set the values for a given row
func setRow(mat *DenseMatrix, which int, row []float64) *DenseMatrix {
	if mat.Cols() != len(row) {
		fmt.Println("The row to set needs to be the same dimension as the matrix")
	}
	// iterate over columns to set the values for a selected row
	for i := 0; i < mat.Cols(); i++ {
		mat.Set(which, i, row[i])
	}
	return mat
}

// a function to set the values for a given column
func setCol(mat *DenseMatrix, which int, col []float64) *DenseMatrix {
	if mat.Rows() != len(col) {
		fmt.Println("The column to set needs to be the same dimension as the matrix")
	}
	// iterate over rows to set the values for a selected columns
	for i := 0; i < mat.Rows(); i++ {
		mat.Set(i, which, col[i])
	}
	return mat
}

// function to substract the minimum from all elements of the matrix
func matrixMinMinus(mat *DenseMatrix) *DenseMatrix {
	values := mat.Array()
	min := float64(100)
	for i := 0; i < len(values); i++ {
		if values[i] < min {
			min = values[i]
		}
	}
	for i := 0; i < len(values); i++ {
		values[i] -= min
	}
	return MakeDenseMatrix(values, mat.Rows(), mat.Cols())
}

// returns the max value of the (dense)matrix
func matrixMax(mat *DenseMatrix) float64 {
	values := mat.Array()
	max := float64(0)
	for i := 0; i < len(values); i++ {
		if values[i] > max {
			max = values[i]
		}
	}
	return max
}

// Alternating Least Squares w/weighting to produce better recommendations
// FOR THE EXPLICIT CASE
// Follows the matrix formulas for X and Y outlined in:
// 				wanlab.poly.edu/recsys12/recsys/p83.pdf
func TrainALS(Q *DenseMatrix, n_factors, iterations int, lambda float64) (*DenseMatrix, float64) {
	W := MakeWeightMatrix(Q)
	maxval := matrixMax(Q)
	X, Y := MakeXY(Q, n_factors, maxval, 47)
	// to store error values
	errors := make([]float64, 0)

	for ii := 0; ii < iterations; ii++ {
		// scaled identity matrix
		I := Eye(n_factors)
		I.Scale(lambda)

		// solve for X
		for u := 0; u < Q.Rows(); u++ {
			weightedRow := W.GetRowVector(u).Array()
			w_yt, _ := Diagonal(weightedRow).TimesDense(Y.Transpose())
			y_wt_yt, _ := Y.TimesDense(w_yt)
			y_wt_yt.AddDense(I)
			y_wt_ytInv, _ := y_wt_yt.Inverse()

			q_u := Q.GetRowVector(u).Transpose()
			wu_qu, _ := Diagonal(weightedRow).TimesDense(q_u)
			x_tosolve, _ := Y.TimesDense(wu_qu)
			new_row, _ := y_wt_ytInv.TimesDense(x_tosolve)
			X = setRow(X, u, new_row.Array())
		}
		// now alternate to solve for Y
		for i := 0; i < Q.Cols(); i++ {
			weightedCol := W.GetColVector(i).Transpose().Array()
			w_x, _ := Diagonal(weightedCol).TimesDense(X)
			x_t_w_x, _ := X.Transpose().TimesDense(w_x)
			x_t_w_x.AddDense(I)
			x_t_w_xInv, _ := x_t_w_x.Inverse()

			q_i := Q.GetColVector(i)
			wi_qi, _ := Diagonal(weightedCol).TimesDense(q_i)
			y_tosolve, _ := X.Transpose().TimesDense(wi_qi)
			new_col, _ := x_t_w_xInv.TimesDense(y_tosolve)
			Y = setCol(Y, i, new_col.Array())
		}
		// Calculate the error values at each iteration
		error_value := getErrorInline(W, Q, X, Y)
		errors = append(errors, error_value)
	}
	fmt.Printf("\nFinal Error value of: %v\n", errors[len(errors)-1])
	weighted_Qhat, _ := X.TimesDense(Y)
	return weighted_Qhat, errors[len(errors)-1]
}

// Alternating Least Squares for the Implicit Case
// Follows the matrix formulas for X and Y outlined in:
// 	'Collaborative Filtering For Implicit Feedback Datasets' by Hu, Koren et al.
//
// C represents the confidence levels derived from the raw observations R, and P is the preferences for values
func TrainALS_Implicit(R *DenseMatrix, n_factors, iterations int, lambda float64) *DenseMatrix {
	P := MakeWeightMatrix(R)
	C := MakeCMatrix(R)
	X, Y := MakeXY(R, n_factors, 5, 47)

	for ii := 0; ii < iterations; ii++ {
		// scaled identity matrix
		I := Eye(n_factors)
		I.Scale(lambda)

		// solve for X
		for u := 0; u < C.Rows(); u++ {
			weightedRow := C.GetRowVector(u).Array()
			c_yt, _ := Diagonal(weightedRow).TimesDense(Y)
			y_ct_yt, _ := Y.Transpose().TimesDense(c_yt)
			y_ct_yt.AddDense(I)
			y_ct_ytInv, _ := y_ct_yt.Inverse()

			p_u := P.GetRowVector(u).Transpose()
			cu_pu, _ := Diagonal(weightedRow).TimesDense(p_u)
			x_tosolve, _ := Y.Transpose().TimesDense(cu_pu)
			new_row, _ := y_ct_ytInv.TimesDense(x_tosolve)
			X = setRow(X, u, new_row.Array())
		}

		// now alternate to solve for Y
		for i := 0; i < C.Cols(); i++ {
			weightedCol := C.GetColVector(i).Transpose().Array()
			c_x, _ := Diagonal(weightedCol).TimesDense(X)
			x_t_c_x, _ := X.Transpose().TimesDense(c_x)
			x_t_c_x.AddDense(I)
			x_t_c_xInv, _ := x_t_c_x.Inverse()

			p_i := P.GetColVector(i)
			ci_pi, _ := Diagonal(weightedCol).TimesDense(p_i)
			y_tosolve, _ := X.Transpose().TimesDense(ci_pi)
			new_col, _ := x_t_c_xInv.TimesDense(y_tosolve)
			Y = setCol(Y, i, new_col.Array())
		}
	}
	weighted_Qhat, _ := X.TimesDense(Y)
	return weighted_Qhat
}

// Returns confidence for a user/product for the Implicit Case.
func Predict(Qhat *DenseMatrix, user, product int) (float64, error) {
	if user > Qhat.Rows() || product > Qhat.Cols() {
		return 0.0, errors.New("User/Product index out of range")
	}
	return Qhat.Get(user, product), nil
}

func oppositeWeights(Q *DenseMatrix) *DenseMatrix {
	mat := Q.Array()
	for i := 0; i < len(mat); i++ {
		if mat[i] > 0 {
			mat[i] = 0
		} else {
			mat[i] = 1
		}
	}
	return MakeDenseMatrix(mat, Q.Rows(), Q.Cols())
}

// looks at the model generated by ALS and makes a user/product prediction
func GetTopNRecommendations(Q, Qhat *DenseMatrix, user, n int, products []string) ([]string, error) {
	qhat := Qhat.Copy()
	inverseWeights := oppositeWeights(Q)
	qhat = simpleTimes(qhat, inverseWeights)

	if user > qhat.Rows() || n > qhat.Cols() {
		return nil, errors.New("User/Product index out of range")
	} else {
		user_row := qhat.GetRowVector(user).Array()
		productScores := make(map[float64]string, 0)
		// make score - product map if product list is present. Else use indices.
		if products != nil {
			for idx, val := range user_row {
				productScores[val] = products[idx]
			}
		} else {
			for idx, val := range user_row {
				productScores[val] = strconv.Itoa(idx)
			}
		}
		// Sort user row
		sort.Sort(sort.Reverse(sort.Float64Slice(user_row)))
		// get top-N recommendations
		var recommendations []string
		for i := 0; i < n; i++ {
			recommendations = append(recommendations, productScores[user_row[i]])
		}
		return recommendations, nil
	}
}
