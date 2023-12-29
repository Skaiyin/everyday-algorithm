use crate::{common::read_data, hmm::HMM};

pub fn forward(data: &Vec<u32>, hmm: &mut HMM) -> Vec<Vec<f64>> {
    // Add your code here
    let len = data.len();
    let state_num = hmm.state_num;
    let mut alpha = vec![vec![0.0; state_num]; len];
    let pi = hmm.initial.clone();
    for i in 0..state_num {
        alpha[0][i] = pi[i] * hmm.observation[0][i];
    }
    for t in 0..len - 1 {
        for j in 0..state_num {
            let mut sum = 0.0;
            for k in 0..state_num {
                sum += alpha[t][k] * hmm.transition[k][j];
            }
            alpha[t + 1][j] = sum * hmm.observation[j][data[t + 1] as usize];
        }
    }
    alpha
}

pub fn backward(data: &Vec<u32>, hmm: &mut HMM) -> Vec<Vec<f64>> {
    // Add your code here
    let len = data.len();
    let state_num = hmm.state_num;
    let mut beta = vec![vec![0.0; state_num]; len];
    for i in 0..state_num {
        beta[len - 1][i] = 1.0;
    }

    for t in (0..len - 1).rev() {
        for i in 0..state_num {
            for j in 0..state_num {
                beta[t][i] += hmm.transition[i][j]
                    * hmm.observation[j][data[t + 1] as usize]
                    * beta[t + 1][j];
            }
        }
    }
    beta
}

pub fn train(path: &str, hmm: &mut HMM) {
    // Add your training code here
    let train_data = read_data(path).unwrap();
    let len = train_data.num_sequence[0].len();
    let num_data = train_data.num_sequence.len();
    let state_num = hmm.state_num;
    let mut gamma = vec![vec![vec![0.0; state_num]; len]; num_data];
    let mut epsilon = vec![vec![vec![vec![0.0; state_num]; state_num]; len - 1]; num_data];

    for (index, data) in train_data.num_sequence.iter().enumerate() {
        let alpha = forward(&data, hmm);
        let beta = backward(&data, hmm);
        for t in 0..len {
            let mut sum = 0.0;
            for j in 0..state_num {
                sum += alpha[t][j] * beta[t][j];
            }
            for i in 0..state_num {
                gamma[index][t][i] = alpha[t][i] * beta[t][i] / sum;
            }
        }

        for t in 0..len - 1 {
            let mut sum = 0.0;
            for i in 0..state_num {
                for j in 0..state_num {
                    sum += alpha[t][i]
                        * hmm.transition[i][j]
                        * hmm.observation[j][data[t + 1] as usize]
                        * beta[t + 1][j];
                }
            }
            for i in 0..state_num {
                for j in 0..state_num {
                    epsilon[index][t][i][j] = alpha[t][i]
                        * hmm.transition[i][j]
                        * hmm.observation[j][data[t + 1] as usize]
                        * beta[t + 1][j]
                        / sum;
                }
            }
        }
    }

    //update
    for i in 0..state_num {
        // initial
        for j in 0..num_data {
            hmm.initial[i] += gamma[j][0][i];
        }
        hmm.initial[i] /= num_data as f64;

        // transition
        for j in 0..state_num {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            for k in 0..num_data {
                for t in 0..len - 1 {
                    sum1 += epsilon[k][t][i][j];
                    sum2 += gamma[k][t][i];
                }
            }
            hmm.transition[i][j] = sum1 / sum2;
        }
        // observation
        for j in 0..hmm.observ_num {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            for k in 0..num_data {
                for t in 0..len {
                    if train_data.num_sequence[k][t] as usize == j {
                        sum1 += gamma[k][t][i];
                    }
                    sum2 += gamma[k][t][i];
                }
            }
            hmm.observation[i][j] = sum1 / sum2;
        }
    }
}
