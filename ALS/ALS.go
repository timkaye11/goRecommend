package main

import (
	"fmt"
	. "github.com/skelterjohn/go.matrix"
	"math"
	"math/rand"
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

// create X and Y matrices for the ALS algorithm
func MakeXY(mat *DenseMatrix, n_factors int) (X, Y *DenseMatrix) {
	rows := mat.Rows()
	cols := mat.Cols()
	X_data := make([]float64, rows*n_factors)
	Y_data := make([]float64, cols*n_factors)
	for i := 0; i < len(X_data); i++ {
		X_data[i] = 5 * rand.Float64()
	}
	for j := 0; j < len(Y_data); j++ {
		Y_data[j] = 5 * rand.Float64()
	}
	X = MakeDenseMatrix(X_data, rows, n_factors)
	Y = MakeDenseMatrix(Y_data, n_factors, cols)
	return
}

// adds up all the elements of the array
func SumMatrix(mat *DenseMatrix) (sum float64) {
	values := mat.Array()
	sum = float64(0)
	for i := 0; i < len(values); i++ {
		sum += values[i]
	}
	return
}

// Auxilliary function for Matrix Solver.
func SwapCols(mat *DenseMatrix, i, j int) *DenseMatrix {
	p := mat.Copy()
	trans := p.Transpose()
	trans.SwapRows(i, j)
	toret := trans.Transpose()
	return toret
}

// solves AX = B using matrix inversion. Utilizes the Solve method from
// Skelter John's matrix package, and creates the X matrix s.t AX = B.
func MatrixSolver(mat, outcome *DenseMatrix) *DenseMatrix {
	rows := make([]float64, 0)
	firstRow, _ := mat.SolveDense(outcome)
	rows = append(rows, firstRow.Array()...)
	for i := 1; i < mat.Cols(); i++ {
		matrix := mat.Copy()
		swapped := SwapCols(outcome, 0, i)
		value, _ := matrix.SolveDense(swapped)
		rows = append(rows, value.Array()...)
	}
	return MakeDenseMatrix(rows, mat.Rows(), mat.Cols())
}

// Scales matrix mat by weight.
func SimpleTimes(mat, weight *DenseMatrix) *DenseMatrix {
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

// Gets the error for the alternating least squares algorithm
func GetErrorInline(W, Q, X, Y *DenseMatrix) float64 {
	dot, err := X.TimesDense(Y)
	errcheck(err)
	err = Q.SubtractDense(dot)
	errcheck(err)
	Prod := SimpleTimes(Q, W)
	tosum := SimpleTimes(Prod, Prod)
	sum := SumMatrix(tosum)
	return sum
}

// Alternating Least Sqaures for Collaborative Filtering w/o weighting.
func ALS_Simple(Q *DenseMatrix, iterations, n_factors int, lambda float64) *DenseMatrix {
	X, Y := MakeXY(Q, n_factors)

	// iterate until convergence
	for i := 0; i < iterations; i++ {
		// save for later use - scaled identity matrix
		I := Eye(n_factors)
		I.Scale(lambda)

		Y_dot, err := Y.TimesDense(Y.Transpose())
		errcheck(err)

		// add scaled identity to dot prod of Y and Y^T
		Y_dot.AddDense(I)
		X_toSolve, _ := Y.TimesDense(Q.Transpose())

		// solve for X
		X = MatrixSolver(Y_dot, X_toSolve)
		X = X.Transpose()

		// Now solve for Y
		X_dot, err := X.TimesDense(X.Transpose())
		errcheck(err)
		X_dot.AddDense(I)
		Y_toSolve, _ := X.TimesDense(Q)
		Y = MatrixSolver(X_dot, Y_toSolve).Transpose()
		X = X.Transpose()
	}
	Q_hat, _ := X.TimesDense(Y)
	return Q_hat
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

// Alternating Least Squares w/weighting to produce better recommendations
// Follows the matrix formulas for X and Y outlined in:
// 				wanlab.poly.edu/recsys12/recsys/p83.pdf
func ALS(Q *DenseMatrix, n_factors, iterations int, lambda float64) *DenseMatrix {
	W := MakeWeightMatrix(Q)
	X, Y := MakeXY(Q, n_factors)
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

			q_u := Q.GetRowVector(u).Transpose()
			wu_qu, _ := Diagonal(weightedRow).TimesDense(q_u)
			x_tosolve, _ := Y.TimesDense(wu_qu)
			new_row, _ := y_wt_yt.Solve(x_tosolve)
			X = setRow(X, u, new_row.Array())
		}

		// now alternate to solve for Y
		for i := 0; i < Q.Cols(); i++ {
			weightedCol := W.GetColVector(i).Transpose().Array()
			w_x, _ := Diagonal(weightedCol).TimesDense(X)
			x_t_w_x, _ := X.Transpose().TimesDense(w_x)
			x_t_w_x.AddDense(I)

			q_i := Q.GetColVector(i)
			wi_qi, _ := Diagonal(weightedCol).TimesDense(q_i)
			y_tosolve, _ := X.Transpose().TimesDense(wi_qi)
			new_col, _ := x_t_w_x.Solve(y_tosolve)
			Y = setCol(Y, i, new_col.Array())
		}
		error_value := GetErrorInline(W, Q, X, Y)
		errors = append(errors, error_value)
	}
	fmt.Printf("Final Error value of: %v", errors[len(errors)-1])
	weighted_Qhat, _ := X.TimesDense(Y)
	return weighted_Qhat
}

// function to substract the minimum from all elements of the matrix
func MatrixMinMinus(mat *DenseMatrix) *DenseMatrix {
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
func MatrixMax(mat *DenseMatrix) float64 {
	values := mat.Array()
	max := float64(0)
	for i := 0; i < len(values); i++ {
		if values[i] > max {
			max = values[i]
		}
	}
	return max
}

// returns the index of the max value of a slice
func argmax(args []float64) (index int) {
	index = 0
	first := 0.0
	for idx, val := range args {
		if val > first {
			index = idx
			first = val
		}
	}
	return
}

// looks at the model generated by ALS and makes predictions/reads the matrix
func Predict(W, Q, Q_hat *DenseMatrix) []int {
	Qhat := Q_hat.Copy()
	Qhat = MatrixMinMinus(Qhat)
	maxRating := MatrixMax(Q)
	qhatMax := float64(maxRating) / MatrixMax(Qhat)
	Qhat.Scale(qhatMax)
	W.Scale(maxRating)
	err := Qhat.SubtractDense(W)
	errcheck(err)

	// find the max value for each row in Q_hat
	max_indices := make([]int, Qhat.Rows())
	for i := 0; i < Qhat.Rows(); i++ {
		i_row := Qhat.GetRowVector(i)
		max_indices[i] = argmax(i_row.Array())
		fmt.Printf("User w/ID: %v will like product w/ID: %v \n", i, max_indices[i])
	}
	return max_indices
}
