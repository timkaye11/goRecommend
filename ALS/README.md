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
import "github.com/timkaye11/goRecommend/ALS"

func main() {
	model := ALS('YOUR_MATRIX_HERE')
}
```