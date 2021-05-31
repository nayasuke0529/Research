library(bayesm)
library(plyr)
library(dplyr)
library(tidyr)
library(rstan)
library(ggmcmc)
library(caret)
library(makedummies)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
options(max.print = 600)

################################################################################
##cofの準備(price)
cof = read.csv("K9N5reg_GL_suq.csv", header = TRUE, fileEncoding = "utf-8")
cof = cof[, c(2:30)]
names(cof)[1] = "hhid"
names(cof)[2] = "choice"


#家計ごとに関する属性データの抽出
demo_cof = read.csv("K9N5regZ.csv", header = TRUE, fileEncoding = "utf-8")
demo_cof = demo_cof[, c(2:9)]
names(demo_cof) = c("hhid", "age", "sex", "CL", "income", "married", "job", "famtype")
demo_cof$income = as.factor(demo_cof$income)
demo_cof$sex = as.factor(demo_cof$sex)
demo_cof$married = as.factor(demo_cof$married)

N_cof <- nrow(cof)
y_cof <- cof$choice

X_cof <- cof %>% select(3:29)
X_cof$intercept = 1

cof_Z <- demo_cof %>% 
  select(-hhid)

hhid_index_cof <- demo_cof %>%
  select(hhid) %>% 
  mutate(ind = seq(1,nrow(demo_cof)))

cof_Z <- data.frame(intercept = rep(1, nrow(cof_Z))) %>% 
  bind_cols(cof_Z)

hhid_cof <- cof %>% 
  select(hhid) %>% 
  left_join(hhid_index_cof)

cof_Z = cof_Z[, c("intercept", "married", "income", "CL", "sex", "age")]
cof_Z = makedummies(cof_Z)
#tmp = dummyVars(~., data=cof_Z)
#cof_Z = as.data.frame(predict(tmp, cof_Z))
#cof_Z = cof_Z[, colnames(cof_Z) != "income.6"]

d.cof = list(NX=nrow(X_cof), NZ=nrow(cof_Z), 
             y=y_cof, X=X_cof, Z=cof_Z, K=9, P1=3, P2=11,
             hhid = hhid_cof$ind)

#d.cof2 = list(NX=nrow(X_cof), NZ=nrow(cof_Z), 
#              y=y_cof, X=X2, Z=cof_Z, K=5, P1=1, P2=6,
#              hhid = hhid_cof$ind)

stan_org = stan_model("suq_GL.stan")
stan_WBIC = stan_model("suq_WBIC.stan")
#stanmodel = stan_model("stan_new_GL.stan")

#=================================
##price_only_ver
fit = sampling(
  stan_org,
  data=d.cof,
  seed=1234,
  chains=4, iter=2000
  )
save.image(file = '2021_0212_K9N5reg_GL_suq.RData')
all(summary(fit)$summary[,"Rhat"] <= 1.10, na.rm=T) #Rhat

get_elapsed_time(fit)

#============================================================================================

fit_WBIC = sampling(
  stan_WBIC,
  data=d.cof,
  seed=1234,
  chains=4, iter=1000)
save.image(file = '2021_0112.RData')
all(summary(fit_WBIC)$summary[,"Rhat"] <= 1.10, na.rm=T) #Rhat

get_elapsed_time(fit2)



#============================================================================================

#plot
p1 = stan_plot(fit,
               pars = c("Delta[8,1]"),
               point_est = "mean",
               show_density = T,
               ci_level = 0.95,
               outer_level = 1.00,
               show_outer_line = T,
               alpha=0.4)+
  ggtitle("The poterior distribution of theta_CL")
plot(p1)

ggmcmc(ggs(fit2), file = '1114.pdf', plot = c('traceplot'))
#=================================

#箱ひげ図
res = data.frame(summary(fit))
res2 = extract(fit, pars=c("beta"))
price = data.frame(res2$beta[, , 1])
price2 = data.frame(res2$beta[, , 2])

write.csv(price, "price.csv")
write.csv(price2, "price2.csv")
temp = res %>%
  select(1, 4:8)
write.csv(temp, "temp_1112.csv")


pin = t(data.frame(extract(fit, pars=c("beta"))$beta[, , 9]))
write.csv(pin, "pin9.csv")





