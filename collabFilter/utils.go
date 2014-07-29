package collabFilter

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	. "github.com/skelterjohn/go.matrix"
)

func max(vals []int) int {
	max := 0
	for _, val := range vals {
		if val > max {
			max = val
		}
	}
	return max
}

func min(vals []int) int {
	min := 10000000
	for _, val := range vals {
		if min > val {
			min = val
		}
	}
	return min
}

// read file with separator and load into a matrix.
// If user/product ID's start at 1, set first product/user at row/col index 0.
// Already tested in ALS package
func Load(path, sep string) *DenseMatrix {
	// read in the file
	f, err := ioutil.ReadFile(path)
	errcheck(err)
	lines := strings.Split(string(f), "\n")

	// determine the number of rows and columns
	col_count := make([]int, 0)
	row_count := make([]int, 0)
	for _, line := range lines {
		values := strings.Split(line, sep)
		if line != "" {
			row, _ := strconv.Atoi(values[0])
			col, _ := strconv.Atoi(values[1])
			row_count = append(row_count, row)
			col_count = append(col_count, col)
		}
	}
	// initialize all values to 0
	mat := Zeros(max(row_count), max(col_count))
	// and set the values accordingly
	for _, line := range lines {
		values := strings.Split(line, sep)
		fmt.Println(values)
		if line != "" {
			row, _ := strconv.Atoi(values[0])
			col, _ := strconv.Atoi(values[1])
			val, _ := strconv.ParseFloat(values[2], 64)
			if min(col_count) == 1 {
				mat.Set(row-1, col-1, val)
			} else {
				mat.Set(row, col, val)
			}
		}
	}
	return mat
}
