### Alternating Least Squares (ALS)

Uses the ALS algorithm outlined in [this article](http://wanlab.poly.edu/recsys12/recsys/p83.pdf)

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
	"github.com/timkaye11/goRecommend/ALS"
	"github.com/skelterjohn/go.matrix"
)

func main() {
	// 0's indicate NA's
	// For this instance, cols indicate movie ID ; rows indicate userID.
	// Explicit Case!
	Q := MakeDenseMatrix([]float64{5, 5, 5, 5, 1,
		0, 0, 0, 4, 1,
		1, 2, 3, 3, 1,
		2, 2, 2, 1, 0,
		5, 2, 5, 1, 0}, 5, 5)

	// Matrix of user/product ratings, number of iterations, number of factors, lambda value
	// Prints final estimated error and returns Qhat, the matrix of predictions. 
	ALS(Q, 1, 5, 0.01)

}
```