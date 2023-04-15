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
zip.dt <- data.table::fread("zip.gz")
zip.some <- zip.dt[1:12]
zip.X.array <- array(
  unlist(zip.dt[1:nrow(zip.dt),-1]),
  c(nrow(zip.dt), zip.size, zip.size, 1))
zip.y.mat <- keras::to_categorical(zip.dt$V1)
colnames(zip.y.mat) <- 0:9
str(zip.y.mat)
zip.some[, observation.i := 1:.N]
zip.some.tall <- zip.some[, {
  data.table::data.table(
    label=V1,
    col.i=rep(1:zip.size, zip.size),
    row.i=rep(1:zip.size, each=zip.size),
    intensity=as.numeric(.SD[, paste0("V", 2:257), with=FALSE]))
}, by=observation.i]

gg.digits <- ggplot()+
  geom_tile(aes(
    x=col.i, y=-row.i, fill=intensity),
    data=zip.some.tall)+
  facet_wrap(observation.i + label ~ ., labeller=label_both)+
  scale_x_continuous(breaks=c(1, 16))+
  scale_y_continuous(breaks=-c(1, 16))+
  coord_equal()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  scale_fill_gradient(low="black", high="white")
png("figure-validation-loss-digits.png",
    width=5.5, height=5, units="in", res=200)
print(gg.digits)
dev.off()

## Linear model.
linear.model <- keras_model_sequential() %>%
  layer_flatten(
    input_shape = dim(zip.X.array)[-1]) %>%
  layer_dense(
    units = ncol(zip.y.mat),
    activation = 'softmax')
str(linear.model)
linear.model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
linear.metrics <- linear.model %>% fit(
  zip.X.array, zip.y.mat,
  batch_size = 100,
  epochs = num_epochs,
  validation_split = 0.2
)
linear.wide <- do.call(data.table::data.table, linear.metrics$metrics)
linear.wide[, epoch := 1:.N]
(linear.tall <- nc::capture_melt_single(
  linear.wide,
  set="val_|", function(x)ifelse(x=="val_", "validation", "subtrain"),
  metric="loss|acc"))
linear.min.loss <- linear.tall[metric=="loss", .SD[which.min(value)], by=set]
linear.min.loss[, loss := "min"]

linear.gg <- ggplot()+
  ggtitle("Linear model")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_line(aes(
    epoch, value, color=set),
    data=linear.tall[metric=="loss"])+
  geom_point(aes(
    epoch, value, color=set, shape=loss),
    data=linear.min.loss)+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
linear.dl <- directlabels::direct.label(linear.gg, "right.polygons")
png("figure-validation-loss-linear.png",
    width=6, height=4, units="in", res=200)
print(linear.dl)
dev.off()

## Deep sparse model.
conv.model <- keras_model_sequential() %>%
  layer_conv_2d(
    input_shape = dim(zip.X.array)[-1],
    filters = 20,
    kernel_size = c(3,3),
    activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = ncol(zip.y.mat), activation = 'softmax')
str(conv.model)
conv.model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
conv.metrics <- conv.model %>% fit(
  zip.X.array, zip.y.mat,
  batch_size = 100,
  epochs = num_epochs,
  validation_split = 0.2
)
conv.wide <- do.call(data.table::data.table, conv.metrics$metrics)
conv.wide[, epoch := 1:.N]
(conv.tall <- nc::capture_melt_single(
  conv.wide,
  set="val_|", function(x)ifelse(x=="val_", "validation", "subtrain"),
  metric="loss|acc"))
conv.min.loss <- conv.tall[metric=="loss", .SD[which.min(value)], by=set]
conv.min.loss[, loss := "min"]

conv.gg <- ggplot()+
  ggtitle("Convolutional neural network")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_line(aes(
    epoch, value, color=set),
    data=conv.tall[metric=="loss"])+
  geom_point(aes(
    epoch, value, color=set, shape=loss),
    data=conv.min.loss)+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
conv.dl <- directlabels::direct.label(conv.gg, "right.polygons")
png("figure-validation-loss-conv.png",
    width=6, height=4, units="in", res=200)
print(conv.dl)
dev.off()

both.tall <- rbind(
  data.table(model="convolutional", conv.tall),
  data.table(model="linear", linear.tall))
both.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ model)+
  geom_line(aes(
    epoch, value, color=set),
    data=both.tall[metric=="loss"])+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
directlabels::direct.label(both.gg, "right.polygons")

## QUIZ 1. purpose of train/subtrain/validation/test
## sets. 2. overfitting/underfitting. 3. data input format for
## ML. 4. cross-validation fold ID / test/train sets.

## deep dense model.
dense.model <- keras_model_sequential() %>%
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
dense.model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
dense.metrics <- dense.model %>% fit(
  zip.X.array, zip.y.mat,
  batch_size = 100,
  epochs = num_epochs,
  validation_split = 0.2
)

dense.wide <- do.call(data.table::data.table, dense.metrics$metrics)
dense.wide[, epoch := 1:.N]
(dense.tall <- nc::capture_melt_single(
  dense.wide,
  set="val_|", function(x)ifelse(x=="val_", "validation", "subtrain"),
  metric="loss|acc"))
dense.min.loss <- dense.tall[metric=="loss", .SD[which.min(value)], by=set]
dense.min.loss[, loss := "min"]
dense.gg <- ggplot()+
  ggtitle("Dense (fully connected) neural network with 8 hidden layers")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_line(aes(
    epoch, value, color=set),
    data=dense.tall[metric=="loss"])+
  geom_point(aes(
    epoch, value, color=set, shape=loss),
    data=dense.min.loss)+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
dense.dl <- directlabels::direct.label(dense.gg, "right.polygons")
png("figure-validation-loss-dense.png",
    width=6, height=4, units="in", res=200)
print(dense.dl)
dev.off()

three.tall <- rbind(
  both.tall,
  data.table(model="dense", dense.tall))
three.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ model)+
  geom_line(aes(
    epoch, value, color=set),
    data=three.tall[metric=="loss"])+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
three.dl <- directlabels::direct.label(three.gg, "right.polygons")
png("figure-validation-loss-three.png",
    width=6, height=4, units="in", res=200)
print(three.gg)
dev.off()
