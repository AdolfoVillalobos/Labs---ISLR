library(tree)

# Carseats Dataset -- Classification

library(ISLR)
attach(Carseats)
High = as.factor(ifelse(Sales<=8, "No", "Yes")) # Transform Sales to a Binary Variable
Carseats = data.frame(Carseats, High)


########### I : Classification Trees

tree.carseats <- tree(High~ .-Sales, Carseats)
summary(tree.carseats)

# Deviance:- 2 Sum_m SUm_l n_mk log p_mk. 
#  n_mk : Number of observations in the m-th terminal nod that belong to the the k_th class
# Small Deviance indicates a tree that provides a good fit to the training data.

plot(tree.carseats)
text(tree.carseats, pretty = 0)


## Estimating the Test Error

set.seed(2)

train = sample(1:nrow(Carseats), 200)
Carseats.test = Carseats[-train, ]
High.test = High[-train]

tree.carseats = tree(High ~ .-Sales, Carseats, subset=train)
tree.pred = predict(tree.carseats, Carseats.test, type="class")
table(tree.pred, High.test)

(104+50)/200

## Prunning

set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN=prune.misclass)
names(cv.carseats)
cv.carseats  # The treee wih size 9 has the lowest CV error.

## Plot the size vs the CV error, and the number of Folds vs the CV error

par(mfrow=c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type ="b")
plot(cv.carseats$k, cv.carseats$dev, type="b")


## Prune the Tree

prune.carseats = prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty=0)

## Perfomance on the Teest Set by the Pruned Tree?

tree.pred = predict(prune.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
(97+58)/200

## Pruned Tree improved interpretability and performance.


###### II : Regression Trees

library(MASS)
set.seed(1)

train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston = tree(medv ~ ., Boston, subset=-train)
summary(tree.boston)

plot(tree.boston)
text(tree.boston, pretty=0)


# Prunning

cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type="b")


prune.boston = prune.tree(tree.boston, best = 6)
plot(prune.boston)
text(prune.boston, pretty=0)


# Prediction on Prunned Tree

yhat = predict(tree.boston, newdata = Boston[-train,])
boston.test = Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat-boston.test)^2)


##### III : Bagging and Random Forest

## Bagging is a special case of RF

library(randomForest)
set.seed(1)
bag.boston = randomForest(medv ~ ., Boston, subset = train, mtry=13, importance=TRUE)
bag.boston

yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)

bag.boston = randomForest(medv ~ ., Boston, subset = train, mtry=13, importance=TRUE, ntree=25)
yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)

## RF
set.seed(1)
mtry = sqrt(13)
rf.boston = randomForest(medv ~ ., Boston, subset = train, mtry=mtry, importance=TRUE)
yhat.rf = predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf-boston.test)^2)

# Importance 

importance(rf.boston)
varImpPlot(rf.boston)


##### IV : Boosting

library(gbm)
set.seed(1)
boost.boston = gbm(medv~., data = Boston[train, ], distribution="gaussian", n.trees=5000, interaction.depth=4)
summary(boost.boston)


par(mfrow=c(1,1)) ## Partial Dependence Plots

plot(boost.boston, i="rm")
plot(boost.boston, i="lstat")


## Prediction

yhat.boost = predict(boost.boston, newdata=Boston[-train, ],
                     n.trees=5000)
mean((yhat.boost-boston.test)^2) ## Similar to Bagging/ RF


## Prediction with Learning Rate
boost.boston = gbm(medv~., data = Boston[train, ],
                   distribution="gaussian", 
                   n.trees=5000,
                   interaction.depth=4,
                   shrinkage = 0.2,
                   verbose=F
                   )
yhat.boost = predict(boost.boston, newdata=Boston[-train, ],
                     n.trees=5000)
mean((yhat.boost-boston.test)^2) #

