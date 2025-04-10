(SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new())
library(data.table)
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/*csv")
##unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/*csv")[1]
task.list <- list()
for(unb.csv in unb.csv.vec){
  data.csv <- sub("_unbalanced", "", unb.csv)
  MNIST_dt <- fread(file=data.csv)
  subset_dt <- fread(unb.csv) 
  subset_dt[, identical(
    seed1_prop0.1=="balanced",
    seed1_prop0.05=="balanced")]
  subset_dt[, identical(
    seed2_prop0.005=="balanced",
    seed2_prop0.001=="balanced")]
  subset_dt[, all(which(
    seed1_prop0.05=="unbalanced"
  ) %in% which(
    seed1_prop0.1=="unbalanced"
  ))]
  subset_dt[, all(which(
    seed2_prop0.001=="unbalanced"
  ) %in% which(
    seed2_prop0.005=="unbalanced"
  ))]
  task_dt <- data.table(subset_dt, MNIST_dt)[, odd := factor(y %% 2)]
  feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
  subset.name.vec <- names(subset_dt)
  subset.name.vec <- c("seed1_prop0.01","seed2_prop0.01")
  (data.name <- gsub(".*/|[.]csv$", "", unb.csv))
  for(subset.name in subset.name.vec){
    subset_vec <- task_dt[[subset.name]]
    task_id <- paste0(data.name,"_",subset.name)
    itask <- mlr3::TaskClassif$new(
      task_id, task_dt[subset_vec != ""], target="odd")
    itask$col_roles$stratum <- "y"
    itask$col_roles$subset <- subset.name
    itask$col_roles$feature <- feature.names
    task.list[[task_id]] <- itask
  }
}
if(FALSE){#verify odd and y distributions.
  SOAK$instantiate(itask)
  SOAK_row <- SOAK$instance$iteration.dt[
    train.subsets=="same" & test.subset=="balanced" & test.fold==1]
  itask$backend$data(SOAK_row$train[[1]], c("odd","y"))[, table(odd, y)]
}
Proposed_AUM <- function(pred_tensor, label_tensor){
  is_positive = label_tensor == 2
  is_negative = !is_positive
  if(all(!as.logical(is_positive)) || all(!as.logical(is_negative))){
    return(torch::torch_sum(pred_tensor*0))
  }
  fn_diff = torch::torch_where(is_positive, -1, 0)
  fp_diff = torch::torch_where(is_positive, 0, 1)
  thresh_tensor = -pred_tensor$flatten()
  sorted_indices = torch::torch_argsort(thresh_tensor)
  fp_denom = torch::torch_sum(is_negative) #or 1 for AUM based on count instead of rate
  fn_denom = torch::torch_sum(is_positive) #or 1 for AUM based on count instead of rate
  sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1)/fp_denom
  sorted_fn_cum = -fn_diff[sorted_indices]$flip(1)$cumsum(dim=1)$flip(1)/fn_denom
  sorted_thresh = thresh_tensor[sorted_indices]
  sorted_is_diff = sorted_thresh$diff() != 0
  sorted_fp_end = torch::torch_cat(c(sorted_is_diff, torch::torch_tensor(TRUE)))
  sorted_fn_end = torch::torch_cat(c(torch::torch_tensor(TRUE), sorted_is_diff))
  uniq_thresh = sorted_thresh[sorted_fp_end]
  uniq_fp_after = sorted_fp_cum[sorted_fp_end]
  uniq_fn_before = sorted_fn_cum[sorted_fn_end]
  FPR = torch::torch_cat(c(torch::torch_tensor(0.0), uniq_fp_after))
  FNR = torch::torch_cat(c(uniq_fn_before, torch::torch_tensor(0.0)))
  roc = list(
    FPR=FPR,
    FNR=FNR,
    TPR=1 - FNR,
    "min(FPR,FNR)"=torch::torch_minimum(FPR, FNR),
    min_constant=torch::torch_cat(c(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(c(uniq_thresh, torch::torch_tensor(Inf))))
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  torch::torch_sum(min_FPR_FNR * constant_diff)
}
if(FALSE){
  w <- torch::nn_linear(2, 1)
  x <- torch::torch_randn(c(3,2))
  pred <- w(x)
  l <- torch::torch_tensor(c(1,1,1))
  aum <- Proposed_AUM(pred,l)
  aum$backward()
}
nn_AUM_loss <- torch::nn_module(
  "nn_AUM_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = Proposed_AUM
)
nn_bce_loss3 = torch::nn_module(
  c("nn_bce_with_logits_loss3", "nn_loss"),
  initialize = function(weight = NULL, reduction = "mean", pos_weight = NULL) {
    self$loss = torch::nn_bce_with_logits_loss(weight, reduction, pos_weight)
  },
  forward = function(input, target) {
    self$loss(input$reshape(-1), target$to(dtype = torch::torch_float())-1)
  }
)
make_torch_learner <- function(id,...){
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(...),
    list(
      mlr3torch::nn("linear", out_features=1),
      mlr3pipelines::po(
        "torch_loss",
        loss_fun),
      mlr3pipelines::po(
        "torch_optimizer",
        mlr3torch::t_opt("sgd", lr=0.1)),
      mlr3pipelines::po(
        "torch_callbacks",
        mlr3torch::t_clbk("history")),
      mlr3pipelines::po(
        "torch_model_classif",
        batch_size = 100000,
        patience=n.epochs,
        measures_valid=measure_list,
        measures_train=measure_list,
        predict_type="prob",
        epochs = paradox::to_tune(upper = n.epochs, internal = TRUE)))
  )
  graph <- Reduce(mlr3pipelines::concat_graphs, po_list)
  glearner <- mlr3::as_learner(graph)
  mlr3::set_validate(glearner, validate = 0.5)
  mlr3tuning::auto_tuner(
    learner = glearner,
    tuner = mlr3tuning::tnr("internal"),
    resampling = mlr3::rsmp("insample"),
    measure = mlr3::msr("internal_valid_score", minimize = TRUE),
    term_evals = 1,
    id=paste0(id,"_",train_loss),
    store_models = TRUE)
}
n.pixels <- 28
n.epochs <- 400
measure_list <- mlr3::msrs(c("classif.auc", "classif.acc"))
mlr3torch_loss_list <- list(
  AUM=nn_AUM_loss,
  logistic=nn_bce_loss3)
learner_list <- list(
  mlr3learners::LearnerClassifCVGlmnet$new(),
  mlr3::LearnerClassifFeatureless$new())
for(train_loss in names(mlr3torch_loss_list)){
  loss_fun <- mlr3torch_loss_list[[train_loss]]
  arch.list <- list(
    make_torch_learner(
      "conv",
      mlr3pipelines::po(
        "nn_reshape",
        shape=c(-1,1,n.pixels,n.pixels)),
      mlr3pipelines::po(
        "nn_conv2d_1",
        out_channels = 20,
        kernel_size = 6),
      mlr3pipelines::po("nn_relu_1", inplace = TRUE),
      mlr3pipelines::po(
        "nn_max_pool2d_1",
        kernel_size = 4),
      mlr3pipelines::po("nn_flatten"),
      mlr3pipelines::po(
        "nn_linear",
        out_features = 50),
      mlr3pipelines::po("nn_relu_2", inplace = TRUE)
    ),
    make_torch_learner("linear"),
    make_torch_learner(
      "dense_50",
      mlr3pipelines::po(
        "nn_linear",
        out_features = 50),
      mlr3pipelines::po("nn_relu_1", inplace = TRUE)
    )
  )
  for(arch.i in seq_along(arch.list)){
    learner_auto <- arch.list[[arch.i]]
    learner_list[[learner_auto$id]] <- learner_auto
  }
}
sapply(learner_list, "[[", "predict_type")
for(learner_i in seq_along(learner_list)){
  learner_list[[learner_i]]$predict_type <- "prob"
}
sapply(learner_list, "[[", "predict_type")
(bench.grid <- mlr3::benchmark_grid(
  task.list,
  learner_list,
  SOAK))
if(FALSE){
  learner_list$linear_AUM$train(task.list[[1]])
  learner_list$linear_AUM$tuning_result$internal_tuned_values
}

reg.dir <- "2025-04-04-nnet_prop0.01"
cache.RData <- paste0(reg.dir,".RData")
if(file.exists(cache.RData)){
  load(cache.RData)
}else{#code below should be run interactively.
  if(on.cluster){
    unlink(reg.dir, recursive=TRUE)
    reg = batchtools::makeExperimentRegistry(
      file.dir = reg.dir,
      seed = 1,
      packages = c("mlr3learners","mlr3torch","glmnet","mlr3resampling")
    )
    mlr3batchmark::batchmark(
      bench.grid, store_models = TRUE, reg=reg)
    job.table <- batchtools::getJobTable(reg=reg)
    chunks <- data.frame(job.table, chunk=1)
    batchtools::submitJobs(chunks, resources=list(
      walltime = 60*60*24,#seconds
      memory = 8000,#megabytes per cpu
      ncpus=1,  #>1 for multicore/parallel jobs.
      ntasks=1, #>1 for MPI jobs.
      chunks.as.arrayjobs=TRUE), reg=reg)
    reg <- batchtools::loadRegistry(reg.dir)
    batchtools::getStatus(reg=reg)
    jobs.after <- batchtools::getJobTable(reg=reg)
    extra.cols <- c(algo.pars="learner_id", prob.pars="task_id")
    for(list_col_name in names(extra.cols)){
      new_col_name <- extra.cols[[list_col_name]]
      value <- sapply(jobs.after[[list_col_name]], "[[", new_col_name)
      set(jobs.after, j=new_col_name, value=value)
    }
    table(jobs.after$error)
    exp.ids <- batchtools::findExpired(reg=reg)
    ids <- jobs.after[is.na(error), job.id]
    ok.ids <- setdiff(ids,exp.ids$job.id)
    keep_history <- function(x){
      learners <- x$learner_state$model$marshaled$tuning_instance$archive$learners
      x$learner_state$model <- if(is.function(learners)){
        L <- learners(1)[[1]]
        x$history <- L$model$torch_model_classif$model$callbacks$history
      }
      x
    }
    bench.result <- mlr3batchmark::reduceResultsBatchmark(ok.ids, fun=keep_history, reg = reg)
  }else{
    ## In the code below, we declare a multisession future plan to
    ## compute each benchmark iteration in parallel on this computer
    ## (data set, learning algorithm, cross-validation fold). For a
    ## few dozen iterations, using the multisession backend is
    ## probably sufficient (I have 12 CPUs on my work PC).
    if(require(future))plan("multisession")
    bench.result <- mlr3::benchmark(bench.grid, store_models = TRUE)
  }
  save(bench.result, file=cache.RData)
}

## Cache results for figures to CSV.
score.csv <- paste0(reg.dir,".csv")
score_dt <- mlr3resampling::score(bench.result, mlr3::msr("classif.auc"))
score_out <- score_dt[, .(
  task_id, test.subset, train.subsets, test.fold, algorithm, classif.auc)]
fwrite(score_out, score.csv)
history_dt <- score_dt[
, learner[[1]]$model
, by=.(task_id, test.subset, train.subsets, test.fold, algorithm)]
history.csv <- paste0(reg.dir,"_history.csv")
fwrite(history_dt, history.csv)

## Read results from CSV to make figures.
library(data.table)
pre <- "2025-04-04-nnet_prop0.01"
score_dt <- fread(paste0(pre,".csv"))
plist <- mlr3resampling::pvalue(score_dt)
plot(plist)  

query <- function(DT)DT[
  grepl("linear",algorithm) & test.subset=="balanced" & train.subsets!="all" & grepl("seed1", task_id)]
plist <- mlr3resampling::pvalue(query(score_dt))
plot(plist)

plist$stats[, let(
  Train = ifelse(Train_subsets=="same","balanced","unbalanced"),
  Data = gsub("_.*", "", task_id)
)]
library(ggplot2)
gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(1, "lines"))+
  geom_text(aes(
    value_mean, algorithm, label=sprintf(
      "%.3f±%.3f", value_mean, value_sd),
    hjust=fcase(
      value_mean<0.8, 0,
      value_mean>0.95, 1,
      default=0.5)),
    vjust=-0.5,
    data=plist$stats)+
  geom_point(aes(
    value_mean, algorithm),
    shape=1,
    data=plist$stats)+
  geom_segment(aes(
    lo, algorithm,
    xend=hi, yend=algorithm),
    data=plist$stats)+
  facet_grid(Train ~ Data, labeller=label_both)+
  scale_x_continuous(
    "Test AUC on balanced subset (mean±SD, 3-fold CV)")
png("2025-04-04-nnet_prop0.01-test-auc.png", width=5, height=3.5, units="in", res=200)
print(gg)
dev.off()

history_dt <- fread(paste0(pre,"_history.csv"))
melt_history <- function(DT)nc::capture_melt_single(
  DT,
  set=nc::alevels(train="subtrain", valid="validation"),
  ".classif.",
  measure=nc::alevels(acc="accuracy_prop", ce="error_prop", auc="AUC", "logloss"))
history_long <- melt_history(history_dt)[
, train.subset := ifelse(train.subsets=="same","balanced","unbalanced")
]
history_show <- query(
  history_long[test.fold==1 & measure=="AUC" & train.subsets!="same"]
)[, let(
  Data = gsub("_.*", "", task_id)
)]
library(ggplot2)
gg <- ggplot()+
  theme_bw()+
  geom_line(aes(
    epoch, value, color=set),
    data=history_show)+
  facet_grid(algorithm ~ Data, labeller=label_both)+
  scale_y_continuous(
    "AUC (unbalanced train set)")
png("2025-04-04-nnet_prop0.01-train-auc.png", width=5, height=3.5, units="in", res=200)
print(gg)
dev.off()
