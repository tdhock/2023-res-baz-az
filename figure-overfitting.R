library(data.table)
library(ggplot2)

n.folds <- 4
unique.folds <- 1:n.folds
sim.data.list <- list()
csv.vec <- Sys.glob("data_*.csv")
for(csv in csv.vec){
  pattern <- gsub("data_|.csv", "", csv)
  sim.dt <-  data.table::fread(csv)
  set.seed(1)
  fold.vec <- sample(rep(unique.folds, l=nrow(sim.dt)))
  sim.data.list[[pattern]] <- data.table(
    pattern, sim.dt, fold=factor(fold.vec))
}
sim.data <- do.call(rbind, sim.data.list)

validation.fold <- 1
sim.data[, set := ifelse(fold==validation.fold, "validation", "subtrain")]
sim.data[, folds := ifelse(
  fold==validation.fold,
  validation.fold,
  paste(unique.folds[-validation.fold], collapse=","))]
sim.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
png("figure-overfitting-data.png", width=5, height=3, units="in", res=200)
print(sim.gg)
dev.off()

folds.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y, color=fold),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
png("figure-overfitting-data-folds.png", width=5, height=3, units="in", res=200)
print(folds.gg)
dev.off()

sets.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(set + folds ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
png("figure-overfitting-data-sets.png", width=5, height=3, units="in", res=200)
print(sets.gg)
dev.off()

nnet.pred.dt.list <- list()
nnet.loss.dt.list <- list()
nnet.err.dt.list <- list()
hidden.units.vec <- c(2, 20, 200)
maxit.vec <- 10^seq(0, 4)
for(pattern in unique(sim.data$pattern)){
  select.pattern <- data.table(pattern)
  pattern.data <- sim.data[select.pattern, on="pattern"]
  set.data <- list()
  for(set.name in unique(pattern.data$set)){
    set.data[[set.name]] <- with(pattern.data[set==set.name], list(x=x, y=y))
  }
  for(hidden.units in hidden.units.vec){
    if(hidden.units==0){
      skip <- TRUE
    }else{
      skip <- FALSE
    }
    for(maxit in maxit.vec){
      cat(paste(pattern, hidden.units, maxit, "\n"))
      set.seed(1)
      fit <- nnet::nnet(
        y ~ x,
        set.data$subtrain,
        size=hidden.units,
        skip=skip,
        linout=TRUE,
        maxit=maxit)
      pred.x <- unique(sort(c(
        sim.data$x,
        seq(min(sim.data$x), max(sim.data$x), l=200)
      )))
      for(set in names(set.data)){
        set.err.dt <- with(set.data[[set]], data.table(
          x, y, pred.y=as.numeric(predict(fit, data.frame(x)))))
        nnet.err.dt.list[[paste(pattern, hidden.units, maxit, set)]] <-
          data.table(pattern, hidden.units, maxit, set, set.err.dt)
      }
      nnet.pred.dt.list[[paste(pattern, hidden.units, maxit)]] <- data.table(
        pattern, hidden.units, maxit,
        x=pred.x,
        pred.y=as.numeric(predict(fit, data.frame(x=pred.x))))
      pattern.data[, pred.y := as.numeric(predict(fit, pattern.data))]
      nnet.loss.dt.list[[paste(pattern, hidden.units, maxit)]] <- data.table(
        pattern, hidden.units, maxit,
        pattern.data[, .(mse=mean((pred.y-y)^2)), by=set])
    }#for(epoch
  }#for(model.name
}#for(pattern)
nnet.pred.dt <- do.call(rbind, nnet.pred.dt.list)
nnet.loss.dt <- do.call(rbind, nnet.loss.dt.list)
folds.info <- unique(sim.data[, .(set, folds)])
nnet.err.dt <- do.call(rbind, nnet.err.dt.list)[folds.info, on="set"]

nnet.min.dt <- nnet.loss.dt[
set=="validation", .SD[which.min(mse)], by=.(hidden.units, set, pattern)]
nnet.min.dt[, loss := "min"]
ggplot()+
  geom_point(aes(
    maxit, mse, color=hidden.units, shape=loss),
    data=nnet.min.dt)+
  geom_line(aes(
    maxit, mse, color=hidden.units, linetype=set),
    data=nnet.loss.dt)+
  scale_x_log10()+
  scale_y_log10()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(hidden.units ~ pattern, labeller=label_both, scales="free")

units.colors <- c(
  ##"#F7FBFF", "#DEEBF7", "#C6DBEF",
  ##"#9ECAE1",
  "2"="#6BAED6", #"#4292C6", 
  "20"="#2171B5", #"#08519C",
  "200"="#08306B")
(valid.only <- nnet.loss.dt[set=="validation"])
(valid.only.min <- valid.only[, .SD[which.min(mse)], by=pattern])
valid.only[, hidden.units.fac := factor(hidden.units)]
gg <- ggplot()+
  geom_line(aes(
    maxit, mse, color=hidden.units.fac),
    size=1,
    data=valid.only)+
  directlabels::geom_dl(aes(
    maxit, mse, label=sprintf(
      "min maxit=%d\nunits=%d", maxit, hidden.units)),
    method=list(cex=0.8, "bottom.polygons"),
    color="white",
    data=valid.only.min)+
  geom_point(aes(
    maxit, mse),
      fill="white",
      shape=21,
    data=valid.only.min)+
  scale_color_manual("hidden\nunits", values=units.colors)+
  scale_x_log10(
    "Maximum number of iterations/epochs in gradient descent learning algorithm",
    limits=valid.only[, c(min(maxit), max(maxit)*5)])+
  scale_y_log10(
    "Mean squared error (one validation set)",
    limits=c(1, max(valid.only$mse))
    )+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)
dl <- directlabels::direct.label(gg, "right.polygons")
png("figure-overfitting-validation-only.png", width=7, height=3, units="in", res=200)
print(dl)
dev.off()

for(units in hidden.units.vec){
  slides.list <- list()
  for(it in maxit.vec){
    gg <- ggplot()+
      ggtitle(paste(
        "Neural network,",
        units,
        "hidden units,",
        it,
        "gradient descent iterations"))+
      theme_bw()+
      theme(panel.spacing=grid::unit(0, "lines"))+
      facet_grid(set + folds ~ pattern, labeller=label_both)+ 
      geom_point(aes(
        x, y),
        shape=1,
        color="grey50",
        data=sim.data)+
      geom_segment(aes(
        x, y,
        xend=x, yend=pred.y),
        data=nnet.err.dt[maxit==it & hidden.units==units])+
      xlab("input/feature x")+
      ylab("output/label y")+
      geom_line(aes(
        x, pred.y),
        color="red",
        data=nnet.pred.dt[maxit==it & hidden.units==units])+
      coord_cartesian(ylim=range(sim.data$y))
    out.png <- sprintf(
      "figure-overfitting-pred-units=%d-maxit=%d.png", units, it)
    print(out.png)
    png(out.png,
        width=7, height=5, units="in", res=200)
    print(gg)
    dev.off()
    slides.list[[length(slides.list)+1]] <- out.png
  }
  gg.units <- ggplot()+
    ggtitle(paste(
      "Neural network,",
      units,
      "hidden units"))+
    geom_point(aes(
      maxit, mse, shape=loss),
      data=nnet.min.dt[hidden.units==units])+
    geom_line(aes(
      maxit, mse, linetype=set),
      data=nnet.loss.dt[hidden.units==units])+
    scale_x_log10(
      "Max iterations of gradient descent learning algorithm",
      limits=c(1, 5e4))+
    scale_y_log10(
      "Mean Squared Error")+
    theme_bw()+
    theme(panel.spacing=grid::unit(0, "lines"))+
    facet_grid(. ~ pattern, labeller=label_both, scales="free")
  out.png <- sprintf("figure-overfitting-data-loss-%d.png", units)
  png(out.png,
      width=7, height=3, units="in", res=200)
  print(gg.units)
  dev.off()
  slides.list[[length(slides.list)+1]] <- out.png
  slides.vec <- sprintf("
\\begin{frame}
  \\includegraphics[width=\\textwidth]{%s}
\\end{frame}
", slides.list)
  writeLines(slides.vec, sprintf("figure-overfitting-%d.tex", units))
}


