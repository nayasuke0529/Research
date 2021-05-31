functions{
  vector Make_prob(int num_p, int num_K, vector beta, row_vector X_s){
    vector[num_p + num_K] vec_beta;
    row_vector[num_p*num_K + 1] vec_X_s;
    vector[num_K] result_vec;
    
    vec_beta = beta;
    vec_X_s = X_s;
    for(i in 1:num_K){
      result_vec[i] = vec_beta[1]*vec_X_s[i]  + vec_beta[2]*vec_X_s[num_K + i] +vec_beta[3]*vec_X_s[num_K*2 + i]  + vec_beta[i + 3]*vec_X_s[num_p*num_K + 1];
    }
    return result_vec;
  }
  matrix kronecker_prod(matrix A, matrix B) {
    matrix[rows(A) * rows(B), cols(A) * cols(B)] C;
    int m;
    int n;
    int p;
    int q;
    m = rows(A);
    n = cols(A);
    p = rows(B);
    q = cols(B);
    for (i in 1:m) {
      for (j in 1:n) {
        int row_start;
        int row_end;
        int col_start;
        int col_end;
        row_start = (i - 1) * p + 1;
        row_end = (i - 1) * p + p;
        col_start = (j - 1) * q + 1;
        col_end = (j - 1) * q + q;
        C[row_start:row_end, col_start:col_end] = A[i, j] * B;
      }
    }
    return C;
  }
}


data{
  int<lower=0> NX;//レコード数
  int<lower=0> NZ;//モニタ数
  int<lower=0> K;//カテゴリ(ブランド)数
  int<lower=0> P1;//説明変数の数(個体内モデル)
  int<lower=0> P2;//説明変数の数（階層モデル）
  
  int y[NX];//選択肢（目的変数）
  matrix[NX, P1*K + 1] X;//説明変数(個体内モデル)
  matrix[NZ, P2 + 1] Z;//説明変数(階層モデル)
  int<lower=0> hhid[NX];//モニタID
}

 
transformed data{
  real nu;
  real<lower=0, upper=0> zero;
  vector[P2 + 1] zeros;
  matrix[P1 + K, P1 + K] I;// 説明変数の数の正方行列
  matrix[P2 + 1, P2 + 1] A;
  
  zeros = rep_vector(0, P2 + 1);
  zero = 0;
  nu = P1 + K + 3; 
  I = diag_matrix(rep_vector(1, P1 + K)); // 1を繰り返しp_x個並べた対角行列を作成
  A = 100*diag_matrix(rep_vector(1, P2 + 1));
}

 
parameters{
  vector[P1 + K - 1] beta_raw[NZ];
  matrix[P2 + 1, P1 + K - 1] Delta_raw;
  cov_matrix[P1 + K] V_b;// 共分散行列
}

 
transformed parameters{
  vector[P1 + K] beta[NZ];
  matrix[P2 + 1, P1 + K]Delta;
  matrix[P1 + K, P1 + K] L_b; //共分散行列
  matrix[(P2 + 1)*(P1 + K), (P2 + 1)*(P1 + K)] L_d; //共分散行列
  
  
  Delta = append_col(Delta_raw, zeros);
  for(i in 1:NZ){
    beta[i] = append_row(beta_raw[i], zero);
  }

  L_b = cholesky_decompose(V_b); // 共分散行列のコレスキー因子をもとめる
  L_d = cholesky_decompose(kronecker_prod(A, V_b)); // 共分散行列に0.01で割ったもののコレスキー因子をもとめる
}

 
model{
  for(i in 1:NX){
    target += categorical_logit_lpmf(y[i] | Make_prob(P1, K, beta[hhid[i]], X[i]));
  }
  for(i in 1:NZ){
    target += multi_normal_cholesky_lpdf(beta[i] | Delta' * Z[i]', L_b);
  }
  target += multi_normal_cholesky_lpdf(to_vector(Delta) | rep_vector(0, (P2 + 1)*(P1 + K)), L_d);
  target += inv_wishart_lpdf(V_b | nu, nu*I);
}

