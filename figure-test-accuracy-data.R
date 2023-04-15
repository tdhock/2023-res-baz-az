library(data.table)
library(ggplot2)
library(keras)
## from https://github.com/rstudio/keras/issues/937
if(FALSE){
  keras::install_keras(version = "2.1.6", tensorflow = "1.5")
}
keras::use_implementation("keras")
keras::use_backend("tensorflow")

num_epochs <- 50
zip.size <- 16
zip <- data.table::fread("zip.gz")
zip.X.mat <- as.matrix(zip[,-1,with=FALSE])
zip.X.array <- array(
  unlist(zip[1:nrow(zip),-1]),
  c(nrow(zip), zip.size, zip.size, 1))
zip.class.tab <- table(zip$V1)
zip.y.mat <- keras::to_categorical(zip$V1, length(zip.class.tab))
str(zip.y.mat)
model.list <- list(
  baseline=NULL,
  linear=keras_model_sequential() %>%
    layer_flatten(
      input_shape = dim(zip.X.array)[-1]) %>%
    layer_dense(
      units = ncol(zip.y.mat),
      activation = 'softmax'),
  conv=keras_model_sequential() %>%
    layer_conv_2d(
      input_shape = dim(zip.X.array)[-1],
      filters = 20,
      kernel_size = c(3,3),
      activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 100, activation = 'relu') %>% 
    layer_dense(units = ncol(zip.y.mat), activation = 'softmax'),
  dense=keras_model_sequential() %>%
    layer_flatten(
      input_shape = dim(zip.X.array)[-1]) %>%
    layer_dense(units = 100, activation = 'relu') %>% 
    layer_dense(units = 100, activation = 'relu') %>% 
    layer_dense(units = 100, activation = 'relu') %>% 
    layer_dense(units = 100, activation = 'relu') %>%
    layer_dense(units = 100, activation = 'relu') %>%
    layer_dense(units = 100, activation = 'relu') %>% 
    layer_dense(units = 100, activation = 'relu') %>%   
    layer_dense(units = 100, activation = 'relu') %>% 
    layer_dense(units = ncol(zip.y.mat), activation = 'softmax')
)

## 4-fold cv.
n.folds <- 4
unique.folds <- 1:n.folds
set.seed(1)
zip[, fold := sample(rep(unique.folds, l=.N))]
table(zip$fold)
cv.results.list <- if(file.exists("figure-test-accuracy-data.rds")){
  readRDS("figure-test-accuracy-data.rds")
}else list()

for(test.fold in unique.folds){
  is.test <- zip$fold == test.fold
  is.train <- !is.test
  train.X.array <- zip.X.array[is.train, , , , drop=FALSE]
  train.y.mat <- zip.y.mat[is.train,]
  test.X.array <- zip.X.array[is.test, , , , drop=FALSE]
  test.y.mat <- zip.y.mat[is.test,]
  for(model.name in names(model.list)){
    fold.model <- paste(test.fold, model.name)
    if(!fold.model %in% names(cv.results.list)){
      cat(sprintf("fold=%d model=%s\n", test.fold, model.name))
      test.list <- if(model.name=="baseline"){
        most.freq.col <- which.max(colSums(train.y.mat))
        list(loss=NA_real_, acc=mean(test.y.mat[, most.freq.col]))
      }else{
        model <- model.list[[model.name]]
        model %>% compile(
          loss = loss_categorical_crossentropy,#for multi-class classification
          optimizer = optimizer_adadelta(),
          metrics = c('accuracy')
        )
        model.metrics <- model %>% fit(
          train.X.array, train.y.mat,
          batch_size = 100,
          epochs = num_epochs,
          validation_split = 0.2,
          verbose=0
        )
        model.wide <- do.call(data.table::data.table, model.metrics$metrics)
        model.wide[, epoch := 1:.N]
        (model.tall <- nc::capture_melt_single(
          model.wide,
          set="val_|", function(x)ifelse(x=="val_", "validation", "subtrain"),
          metric="loss|acc"))
        model.min.loss <- model.tall[
          metric=="loss", .SD[which.min(value)], by=set]
        model.min.loss[, loss := "min"]
        ## plot loss curves for subtrain/validation sets.
        model.gg <- ggplot()+
          theme_bw()+
          theme(panel.spacing=grid::unit(0, "lines"))+
          geom_line(aes(
            epoch, value, color=set),
            data=model.tall[metric=="loss"])+
          geom_point(aes(
            epoch, value, color=set, shape=loss),
            data=model.min.loss)+
          scale_y_log10("loss value (lower for better predictions)")+
          scale_x_continuous(
            "epoch (gradient descent passes through subtrain data)",
            limits=c(0, num_epochs*1.2),
            breaks=seq(0, num_epochs, by=10))
        directlabels::direct.label(model.gg, "right.polygons")
        ##re-train on full train set with best epochs on validation.
        best_epochs <- model.min.loss[set=="validation", epoch]
        model.metrics <- model %>% fit(
          train.X.array, train.y.mat,
          batch_size = 100,
          epochs = best_epochs,
          validation_split = 0,
          verbose=0
        )
        ## compute test accuracy.
        evaluate(model, test.X.array, test.y.mat)
      }
      cv.results.list[[fold.model]] <- with(test.list, data.table(
        test.fold, model.name, loss, acc))
    }#if no cached
  }#for(model.name
}#for(test.fold
saveRDS(cv.results.list, "figure-test-accuracy-data.rds")
