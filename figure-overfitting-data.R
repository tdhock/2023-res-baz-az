true.f.list <- list(
  linear=function(x)2*x + 5,
  quadratic=function(x)x^2,
  sin=function(x)5*sin(2*x)+5)
set.seed(1)
N <- 100
x <- runif(N, -3, 3)
for(pattern in names(true.f.list)){
  true.f <- true.f.list[[pattern]]
  set.seed(1)
  y <- true.f(x) + rnorm(N, 0, 2)
  sim.dt <- data.table::data.table(x, y)
  data.table::fwrite(sim.dt, sprintf("data_%s.csv", pattern))
}
