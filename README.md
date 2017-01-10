This post is a brief example of a few things that I've been working on - topic modeling (particularly on social media data) and a bit of basic clustering. This uses a package that I wrote to simplify topic modeling using Latent Dirichlet Allocation called `litMagModelling`, which can be installed using `devtools`. The data I take from Albert Y. Kim's's repository `JSE_OkCupid`. 

In order to get everything working, we first have to increase Java's available memory size.
```
options(java.parameters = "-Xmx4g")
```
I recommend just downloading the data directly from the url below as it's significantly faster, but if you'd rather I've provided the instructions that Albert does below.
```
# download instructions as per rudeboybert's github
# this takes forever to run on my machine; I think it's easier and faster to just download it manually
url <- "https://github.com/rudeboybert/JSE_OkCupid/blob/master/profiles.csv.zip?raw=true"
temp_zip_file <- tempfile()
download.file(url, temp_zip_file)
unzip(temp_zip_file, "profiles.csv")
# Load CSV into R:
profiles <- read.csv(file="profiles.csv", header=TRUE, stringsAsFactors = FALSE)
```

Let's first combine all of the individual essays into a single meta-essay. This will give us a better picture of what each individual is writing about as a whole, but loses some information - we lose the context that each essay occurs in (the topic of each essay). Another approach would be to run LDA on each of these essays individually with far fewer topics per essay, and then aggregate up (probably through clustering). For now though, let's go with the simple approach.
```
library(tidyverse)
# some brief cleaning
text_cleaning <- function(dataframe){
  dataframe[] <- lapply(dataframe, function(x){
    gsub("\n", " ", x)
  })
  dataframe[] <- lapply(dataframe, function(x){
    gsub("<br />", " ", x)
  })
  dataframe
}
profiles[,7:16] <- text_cleaning(profiles[,7:16])
# probably a cleaner way to do this, but oh well
profiles <- mutate(profiles, essay_full = paste(essay0, essay1, essay2, essay3, essay4, essay5, essay6, essay7, essay8, essay9))
library(devtools)
install_github("mlinegar/litMagModelling")
library(litMagModelling)
library(tm)
# unfortunately, we have to create a stopword file, as this is the format that MALLET takes
# note that MALLET was chosen because it performed significantly faster during testing
# as compared to other tested options
# for more validity we might construct a collection of phrases (so we'd have "monty_python" instead of two separate words)
stopListVec <- c("a", "an", "the", "and", "of", "to", "that", "if", "in", "to", "href", "or", "with", 
                 "have", "it", "is", "ilink", "amp", "by", letters, LETTERS, "www", "com", "http", "https",
                 "ve", "nt", "re", "em", "nbsp", "san", "at", "ll", "on", "gt", tm::stopwords("english"))
fileConn <- file("stopList.txt")
writeLines(stopListVec, fileConn)
close(fileConn)
# we also need to make an ID column
profiles <- mutate(profiles, id = rownames(profiles))
```
Now we can run LDA. While a little much, let's try 100 topics. This takes around 18 minutes on my 2014 Macbook Air with 4g RAM (or around 12 minutes for 20 topics). 
```
ntopics <- 100
profiles_model <- make_model(profiles, n.topics = ntopics, textcolname = "essay_full", stopListFile = "stopList.txt")
# note that we must match n.topics between these two lines
profiles_model.df <- modelToDataframe(profiles_model, profiles, n.topics = ntopics, num.topwords = 10, textcolname = "essay_full")
```
Now that we've made our model (a quick note about just how easy that was - it took 2 lines!), we can begin clustering. Let's start with a basic k-means cluster. We'll set K to be an arbitrary number at first - let's say 10. 
```
library(cluster)
example_cluster <- kmeans(profiles_model.df[,34:ncol(profiles_model.df)], 10, 1000, algorithm="MacQueen")
# this is a little hard to see, but clearly we're not explaining too much of the variability with these clusters
clusplot(profiles_model.df[,34:ncol(profiles_model.df)], example_cluster$cluster, color = TRUE, shade = TRUE)
```
![Alt](Clusterplot_10clusters.pdf "Clusterplot on OKCupid profiles with 10 clusters")

Rather than just guessing how the number of clusters to use, we can use an "elbow plot", which attempts to find an ideal tradeoff between within-cluster sum of squares and the nunmber of clusters. I took this code from [here][1], but there are many resources that focus on different measures, in particular, minimizing negative log likelihood. 
```
# Determine number of clusters
# Calculating K-means – Cluster assignment & cluster group steps
cost_df <- data.frame()
for(i in 1:150){
  kmeans <- kmeans(x=profiles_model.df[,34:ncol(profiles_model.df)], centers=i, iter.max=1500, algorithm="MacQueen")
  cost_df<- rbind(cost_df, cbind(i, kmeans$tot.withinss))
}
names(cost_df) <- c("cluster", "cost")
#Elbow method to identify the ideal number of clusters
#Cost plot
library(ggplot2)
ggplot(data=cost_df, aes(x=cluster, y=cost, group=1)) +
  theme_bw(base_family="Helvetica") +
  geom_line(colour = "darkgreen") +
  theme(text = element_text(size=20)) +
  ggtitle("Reduction In Cost For Values of 'k'") +
  xlab("Clusters") +
  ylab("Within-Cluster Sum of Squares")
```
![Alt](elbowplot.pdf "Elbow Plot")

It looks like 25-30 minimizes within-cluster sum of squares and the number of clusters. Let's try graphing 30 clusters to see if our performance has improved. Also note that we've switched to the MacQueen algorithm for K-means clustering, as I've found that it has better performance for this particular dataset (the default algorithm doesn't converge for 1-2000 iterations).
```
# let's make the points transparent so it's easier to see more dense areas
# play around with the alpha value to change transparency
library(grDevices)
transp_blue <- rgb(0, 0, 255, max = 255, alpha = 5, names = "transparent_blue")
example_cluster30 <- kmeans(profiles_model.df[,34:ncol(profiles_model.df)], 30, 1000, algorithm="MacQueen")
clusplot(profiles_model.df[,34:ncol(profiles_model.df)], example_cluster30$cluster, color = TRUE,shade = TRUE, col.p = transp_blue, main = "CLUSPLOT of topic-makeup of OKCupid profiles")
```
![Alt](Clusterplot_30clusters.pdf "Clusterplot on OKCupid profiles with 30 clusters")

Our first two principle components still explain less than 5% of our varition. Let's try exploring other clustering/aggregation methods. Due to the size of this data (around 60,000 profiles) hierarchical clustering (which initially assigns each point to its own cluster) is impractical. Instead let's try using a "clustergram", as done [here][2]. Note that this code can be found on the other file in the repo. 

Normally we'd scale the data, but I'll leave out that step for now, as all of the data is within (0,1). The basic intuition for this plot is that it allows us to see how particular users move between clusters as we change the total number of clusters. If users remain within a particular group even as the number of clusters increases, that suggests that those groups are in fact distinct. Let's run this clustergram on 20-30 clusters, the approximate range that our elbow plot suggested was ideal. This takes quite a while (around 4 hours on my machine). 
```
par(cex.lab = 1.5, cex.main = 1.2)
profile_matrix <- as.matrix(profiles_model.df[,34:ncol(profiles_model.df)])
clustergram(profile_matrix, k.range = 20:30, line.width = 0.00000001, iter.max = 3000, nstart = 25)
```
![Alt](clustergram.pdf "Clustergram on OKCupid profiles for 20-30 clusters")

As we can see, there are at least 3 groups that remain distinct from the pack. In further analysis we should examine who belongs to each of these groups, and determine whether they are just talking about different topics, or whether there are demographic differences between them.

[1]: https://www.r-bloggers.com/cluster-analysis-using-r/ "Elbow Plot Code"
[2]: https://www.r-bloggers.com/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/ "Clustergram source code"