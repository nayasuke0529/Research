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
  int<lower=0> NX;//���R�[�h��
  int<lower=0> NZ;//���j�^��
  int<lower=0> K;//�J�e�S��(�u�����h)��
  int<lower=0> P1;//�����ϐ��̐�(�̓����f��)
  int<lower=0> P2;//�����ϐ��̐��i�K�w���f���j
  
  int y[NX];//�I�����i�ړI�ϐ��j
  matrix[NX, P1*K + 1] X;//�����ϐ�(�̓����f��)
  matrix[NZ, P2 + 1] Z;//�����ϐ�(�K�w���f��)
  int<lower=0> hhid[NX];//���j�^ID
}

 
transformed data{
  real nu;
  real<lower=0, upper=0> zero;
  vector[P2 + 1] zeros;
  matrix[P1 + K, P1 + K] I;// �����ϐ��̐��̐����s��
  matrix[P2 + 1, P2 + 1] A;
  
  zeros = rep_vector(0, P2 + 1);
  zero = 0;
  nu = P1 + K + 3; 
  I = diag_matrix(rep_vector(1, P1 + K)); // 1���J��Ԃ�p_x���ׂ��Ίp�s����쐬
  A = 100*diag_matrix(rep_vector(1, P2 + 1));
}

 
parameters{
  vector[P1 + K - 1] beta_raw[NZ];
  matrix[P2 + 1, P1 + K - 1] Delta_raw;
  cov_matrix[P1 + K] V_b;// �����U�s��
}

 
transformed parameters{
  vector[P1 + K] beta[NZ];
  matrix[P2 + 1, P1 + K]Delta;
  matrix[P1 + K, P1 + K] L_b; //�����U�s��
  matrix[(P2 + 1)*(P1 + K), (P2 + 1)*(P1 + K)] L_d; //�����U�s��
  
  
  Delta = append_col(Delta_raw, zeros);
  for(i in 1:NZ){
    beta[i] = append_row(beta_raw[i], zero);
  }

  L_b = cholesky_decompose(V_b); // �����U�s��̃R���X�L�[���q�����Ƃ߂�
  L_d = cholesky_decompose(kronecker_prod(A, V_b)); // �����U�s���0.01�Ŋ��������̂̃R���X�L�[���q�����Ƃ߂�
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

