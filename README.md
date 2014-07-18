### Go Recommend

> Recommendation algorithms (Collaborative Filtering)  w/Go! 

Collaborative Filtering (CF) is oftentimes used for item recommendations for users, and many libraries exist for other languages (popular implementations include Mahout, Prediction.IO, Apache MLLib ALS etc..). While many approaches to collaborative filtering and recommendation build off user similarity (either through cosine similarity or correlation similarity), there are also several algorithms that are more complex. As there are very few machine learning packages out there for [Go](http://www.golang.org), I decided to put together some CF algorithms that I thought were interesting. Programming in Go has many benefits - namely speed/efficiency, but also the bare-bones nature allows for more customization and makes ML programming less black-box-y. 

### Collaborative Filters inside this package


- Simple Bayesian CF Algorithm <- inside bayesianFilter/
- [Alternating Least Squares](http://labs.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf) and Weighted Alternating Least Squares (Both for the Implicit and Explicit Case)
	* Currently only the Explicit case is available
	* Tests not yet complete
- Similarity (using correlation, cosine and jaccard similarity) based CF, which incorporates a nearest neighbor type metric can be found in the CF folder.
	* Not complete yet. Soon to come! 

*Most* of the recommendation algorithms to date are outlined in [this article](http://www.hindawi.com/journals/aai/2009/421425/)


#### Additional

If you have any questions/comments, *please* feel free to reach me at tim.kaye@lytics.io




