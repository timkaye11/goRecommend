### Alternating Least Squares (ALS)

> Uses the ALS algorithm outlined in [this article](http://wanlab.poly.edu/recsys12/recsys/p83.pdf) for the explicit case
For the implicit case, the algorithm is outlined [here](labs.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf)

Relies on Skelter John's *matrix.go* package for some matrix functionality. 

#### Usage

1. Install
```go get github.com/timkaye11/goRecommend/ALS```

2. Download the dependencies
	 - If [GPM](http://www.github.com/pote/gpm) is not installed, then do:
	 ``` brew install gpm```
	 - Get the dependencies:
	 ``` gpm install```

3. Run Code
``` 
package main

import (
	. "github.com/timkaye11/goRecommend/ALS"
	"github.com/skelterjohn/go.matrix"
)

func main() {
	// 0's indicate NA's
	// For this instance, cols indicate movie ID ; rows indicate userID.
	Q := MakeDenseMatrix([]float64{5, 5, 5, 5, 1,
		0, 0, 0, 4, 1,
		1, 2, 3, 3, 1,
		2, 2, 2, 1, 0,
		5, 2, 5, 1, 0}, 5, 5)

	// Matrix of user/product ratings, number of iterations, number of factors, lambda value
	// Prints final estimated error and returns Qhat, the matrix of predictions. 
	// This is for the explicit case
	trainALS(Q, 10 , 5, 0.01)


	// For the implicit case, doesn't return error values
	// Returns a confidence score for each entry from [0, 1)
	trainALS_Implicit(Q, 10, 5, 0.01)


}
```