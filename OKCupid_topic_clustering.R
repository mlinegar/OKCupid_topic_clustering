# increase Java's available memory size
options(java.parameters = "-Xmx4g")
# download instructions as per rudeboybert's github
# this takes forever to run on my machine; I think it's easier and faster to just download it manually
# as such, it's currently commented out
# url <- "https://github.com/rudeboybert/JSE_OkCupid/blob/master/profiles.csv.zip?raw=true"
# temp_zip_file <- tempfile()
# download.file(url, temp_zip_file)
# unzip(temp_zip_file, "profiles.csv")
# Load CSV into R:
profiles <- read.csv(file="profiles.csv", header=TRUE, stringsAsFactors = FALSE)
# first we'll combine all of the essays into a single meta-essay
# this will give us a better picture of what individuals are doing as a whole
# however, it removes essay-specific information that we might want
# another approach might be to run LDA on each of the essays individually for a smaller number of topics
# and then aggregate up (cluster) from there
# for now, though, let's stick with the simple approach
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
# takes around 12 minutes on my 2014 Macbook Air with 4g RAM for 20 topics
# or 18 minutes for 100 topics
ntopics <- 100
profiles_model <- make_model(profiles, n.topics = ntopics, textcolname = "essay_full", stopListFile = "stopList.txt")
# note that we must match n.topics between these two lines
profiles_model.df <- modelToDataframe(profiles_model, profiles, n.topics = ntopics, num.topwords = 10, textcolname = "essay_full")

# now let's cluster!
# we can start with a basic k-means cluster - we'll set an arbitrary number at first, say, 10
library(cluster)
example_cluster <- kmeans(profiles_model.df[,34:ncol(profiles_model.df)], 10, 1000, algorithm="MacQueen")
# this is a little hard to see, but clearly we're not explaining too much of the variability with these clusters
clusplot(profiles_model.df[,34:ncol(profiles_model.df)], example_cluster$cluster, color = TRUE, shade = TRUE)
# now let's try using the "elbow method"; I took this code from below:
# https://www.r-bloggers.com/cluster-analysis-using-r/

# Determine number of clusters
# Calculating K-means – Cluster assignment & cluster group steps
cost_df <- data.frame()
for(i in 1:150){
  kmeans<- kmeans(x=profiles_model.df[,34:ncol(profiles_model.df)], centers=i, iter.max=1500, algorithm="MacQueen")
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

# it looks like 25-30 minimizes within-cluster sum of squares and the number of clusters
# let's make the points transparent so it's easier to see more dense areas
# play around with the alpha value to change transparency
library(grDevices)
transp_blue <- rgb(0, 0, 255, max = 255, alpha = 5, names = "transparent_blue")
example_cluster30 <- kmeans(profiles_model.df[,34:ncol(profiles_model.df)], 30, 1000, algorithm="MacQueen")
clusplot(profiles_model.df[,34:ncol(profiles_model.df)], example_cluster30$cluster, color = TRUE, 
         shade = TRUE, col.p = transp_blue, main = "CLUSPLOT of topic-makeup of OKCupid profiles")
# still, this isn't a very good visual representation, as these components explain less than
# 5% of our variation
# due to the size of the data, using hierarchical clustering (which initially assigns each point to its own cluster)
# is impractical - let's try using a "clustergram", as done here:
# https://www.r-bloggers.com/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/
# for now I won't scale the data, we'll check later to see if it matters
par(cex.lab = 1.5, cex.main = 1.2)
profile_matrix <- as.matrix(profiles_model.df[,34:ncol(profiles_model.df)])
# the basic intuition for this plot is that it allows us to see how single points
# move from cluster to cluster as the number of clusters increases. 
# we hope for things to get relatively stable near the optimum
# I chose 20-30 clusters as that's what our above "elbow plot" suggested was ideal
# at low iter.max values we don't have convergence
# takes quite a while - around 4 hours
# we see that there's one group that's consistantly separate from the others
start.time <- Sys.time()
clustergram(profile_matrix, k.range = 20:30, line.width = 0.00000001, iter.max = 3000, nstart = 25)
end.time <- Sys.time()
total.time <- end.time - start.time
total.time
# let's retry and see if scaling actually matters
profile_matrix_scaled <- scale(profile_matrix)
# we do need to have a much higher iter.max now
# might also/alternatively have to change the algorithm we use
start.time <- Sys.time()
clustergram(profile_matrix_scaled, k.range = 20:30, line.width = 0.00000001, iter.max = 3000, nstart = 25)
end.time <- Sys.time()
total.time <- end.time - start.time
total.time

# haven't been able to get this working - oh well
# let's also explore Mclust() which provides an optimal number of clusters by minimizing BIC
library(mclust)
start.time <- Sys.time()
fit <- Mclust(profiles_model.df[,34:ncol(profiles_model.df)], modelNames = mclust.options("emModelNames"))
plot(fit)
summary(fit)
end.time <- Sys.time()
total.time <- end.time - start.time