use crate::common::read_data;
use crate::hmm::HMM;
use crate::train::forward;
pub fn test(test_path: &str, hmm: &mut HMM) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let test_data = read_data(test_path).unwrap();
    let mut delta = vec![vec![0.0; hmm.state_num]; test_data.num_sequence.len()];
    let mut res = Vec::new();
    for data in test_data.num_sequence.iter() {
        delta = forward(&data, hmm);
        let pro= delta[delta.len() - 1].iter().sum();
        res.push(pro);
    }
    Ok(res)
}
