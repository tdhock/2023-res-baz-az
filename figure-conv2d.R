library(ggplot2)
library(data.table)
n.pixels <- 6
pixel.seq <- 1:n.pixels
node.dt <- CJ(x=pixel.seq, y=pixel.seq)
out.node.dt.list <- list()
edge.dt.list <- list()
n.filters <- 2
kernel.size.vec <- c(2,3)
for(filter.i in 1:n.filters){
  x.out <- n.pixels+1+filter.i
  for(kernel.size in kernel.size.vec){
    weights.per.filter <- kernel.size^2
    stride <- kernel.size
    first.seq <- seq(1, n.pixels, by=kernel.size)
    first.dt <- CJ(x=first.seq, y=first.seq)[, y.out := (1:.N)-0.5]
    outputs.per.filter <- nrow(first.dt)
    edge.dt.list[[paste(kernel.size, x.out)]] <- first.dt[, {
      offset <- seq(0, kernel.size-1)
      edges <- CJ(x.in=x+offset, y.in=y+offset)[, weightID := (1:.N)+9*filter.i]
      data.table(
        kernel.size, filter.i, x.out, 
        weights.per.filter, outputs.per.filter, stride,
        edges)
    }, by=y.out]
    out.node.dt.list[[paste(kernel.size, x.out)]] <- data.table(
      kernel.size, filter.i, x.out, 
      weights.per.filter, outputs.per.filter, stride,
      first.dt)
  }
}
out.node.dt <- rbindlist(out.node.dt.list)
edge.dt <- rbindlist(edge.dt.list)
uid <- unique(edge.dt$weightID)
set.seed(1)
new.uid <- structure(sample(uid), names=uid)
edge.dt[, WeightID := factor(new.uid[paste(weightID)]) ]
node.size <- 10
node.fill="white"
ggplot()+
  facet_grid(filter.i ~ kernel.size + weights.per.filter + outputs.per.filter, labeller=label_both)+
  geom_point(aes(
    x, y), 
    data=node.dt,
    size=node.size,
    shape=21)+
  geom_point(aes(
    x.out, y.out),
    data=out.node.dt,
    size=node.size,
    shape=21)+
  geom_segment(aes(
    x.in, y.in,
    color=WeightID,
    xend=x.out, yend=y.out),
    data=edge.dt)+
  coord_equal()+
  scale_x_continuous(
    breaks=seq(1,100))+
  scale_y_continuous(
    breaks=seq(1,100))
for(ksize in kernel.size.vec){
  kernel.edges <- edge.dt[kernel.size==ksize]
  kernel.out <- out.node.dt[kernel.size==ksize]
  meta <- kernel.out[1]
  gg <- ggplot()+
    meta[, ggtitle(sprintf("%dx%d pixel image input, convolutional kernel size = stride = %d,\nn.filters/channels = %d, weights per channel = %d, outputs per channel = %d",n.pixels,n.pixels,kernel.size,n.filters,weights.per.filter,outputs.per.filter))]+
    facet_grid(. ~ filter.i, labeller=label_both)+
    geom_point(aes(
      x, y), 
      data=node.dt,
      size=node.size,
      fill=node.fill,
      shape=21)+
    geom_point(aes(
      x.out, y.out),
      data=kernel.out,
      size=node.size,
      fill=node.fill,
      shape=21)+
    geom_segment(aes(
      x.in, y.in,
      color=WeightID,
      xend=x.out, yend=y.out),
      data=kernel.edges)+
    coord_equal()+
    geom_text(aes(
      x, y, hjust=hjust, label=label),
      vjust=1,
      data=data.table(y=0.1, rbind(
        data.table(x=n.pixels, hjust=1, label="inputs"),
        data.table(x=n.pixels+2, hjust=0, label="outputs"))))+
    scale_x_continuous(
      breaks=seq(1,100))+
    scale_y_continuous(
      limits=c(0,9),
      breaks=seq(1,100))+
    theme(legend.position = "none")
  out.png <- sprintf("figure-conv2d-%dx%d-kernel=%d.png", n.pixels,n.pixels,meta$kernel.size)
  png(out.png, width=8, height=5, units="in", res=200)
  print(gg)
  dev.off()
}
