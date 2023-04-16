library(data.table)
library(ggplot2)
cv.results.list <- readRDS("figure-test-accuracy-data.rds")
cv.results <- do.call(rbind, cv.results.list)
off <- 0.3
model.vjust <- c(
  conv=1+off,
  dense=1+off,
  linear=1+off)
model.hjust <- c(
  conv=1,
  dense=0.5,
  linear=0.5)
cv.stats <- cv.results[, .(
  mean.accuracy=mean(acc),
  median=median(acc),
  max=max(acc),
  min=min(acc),
  q25=quantile(acc, 0.25),
  q75=quantile(acc, 0.75),
  sd.accuracy=sd(acc)
), by=.(
  model.name,
  vjust=model.vjust[model.name],
  hjust=model.hjust[model.name]
)][order(median)]
mfac <- function(x)factor(x, cv.stats$model.name)
cv.results[, Model := mfac(model.name)]
cv.stats[, Model := mfac(model.name)]
cv.stats[, text.x := if(Model=="conv")q75 else median, by=Model]
gg.baseline <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_point(aes(
    acc, Model),
    data=cv.results)+
  scale_x_continuous(
    "Accuracy rate (correctly predicted labels, four test folds)")
png("figure-test-accuracy-baseline.png",
    width=5, height=1.5, units="in", res=200)
print(gg.baseline)
dev.off()
long.panel <- c(
  folds="Zoom out to include baseline
four test folds, mean +/- sd",
stats="Zoom in to best models
25% quantile ( median ) 75% quantile")
pfac <- function(x)factor(long.panel[x], long.panel)
some.stats <- cv.stats[Model != "baseline"]
cv.results[, panel := pfac("folds")]
cv.stats[, panel := pfac("folds")]
off <- 0.05
cv.stats[, text.x := if(Model=="baseline")max+off else min-off, by=Model]
cv.stats[, hjust := if(Model=="baseline")0 else 1, by=Model]
some.stats[, panel := pfac("stats")]
stat.size <- 3
cv.wide <- dcast(cv.results, test.fold ~ Model, value.var="acc")
test.dt.list <- list()
for(larger.i in 3:length(cv.wide)){
  smaller.i <- larger.i-1
  test.result <- t.test(
    cv.wide[[smaller.i]], cv.wide[[larger.i]], "less", paired = TRUE)
  test.dt.list[[paste(larger.i)]] <- data.table(
    smaller.Model=mfac(names(cv.wide)[[smaller.i]]),
    larger.Model=mfac(names(cv.wide)[[larger.i]]),
    p.value=test.result[["p.value"]])
}
test.dt <- do.call(rbind, test.dt.list)
gg.both <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ panel, scales="free")+
  geom_blank(aes(
    x, y),
    data=data.table(x=0, y=mfac("baseline"), panel=pfac("folds")))+
  geom_point(aes(
    acc, Model),
    shape=1,
    data=cv.results)+
  scale_y_discrete("Learned function", drop=FALSE)+
  scale_x_continuous(
    "Accuracy rate (correctly predicted labels/observations)")+
  geom_segment(aes(
    q25, Model,
    xend=q75, yend=Model),
    data=some.stats)+
  geom_point(aes(
    median, Model),
    data=some.stats)+
  geom_text(aes(
    text.x, Model,
    hjust=hjust,
    label=sprintf("%.3f +/- %.3f", mean.accuracy, sd.accuracy)),
    size=stat.size,
    data=cv.stats)+
  geom_text(aes(
    text.x, Model,
    hjust=hjust,
    vjust=vjust,
    label=sprintf("%.3f ( %.3f ) %.3f", q25, median, q75)),
    size=stat.size,
    data=some.stats)
png("figure-test-accuracy-both.png",
    width=5.5, height=1.8, units="in", res=300)
print(gg.both)
dev.off()

gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_segment(aes(
    q25, Model,
    xend=q75, yend=Model),
    data=some.stats)+
  geom_point(aes(
    median, Model),
    data=some.stats)+
  scale_x_continuous(
    "Accuracy rate (correctly predicted labels, median and quartiles)")
png("figure-test-accuracy.png", width=5, height=1.5, units="in", res=200)
print(gg)
dev.off()

