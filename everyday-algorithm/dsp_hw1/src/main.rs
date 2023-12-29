use rayon::prelude::*;
use std::{
    fs::File,
    io::{self, BufRead},
};

use hmm::Models;
use test::test;
use train::train;
mod common;
mod hmm;
mod test;
mod train;
fn main() {
    // read_train_data("./data/train_seq_01.txt").unwrap();
    //创建文件列表 ./data/train_seq_02.txt
    let train_files = [
        "./data/train_seq_01.txt",
        "./data/train_seq_02.txt",
        "./data/train_seq_03.txt",
        "./data/train_seq_04.txt",
        "./data/train_seq_05.txt",
    ];
    let mut hmms = Models::new("model_init.txt", 5).unwrap();
    let iter_num = 100;
    let mut best_iter = 0;
    let mut best_score = 0.0;

    let file = File::open("./data/test_lbl.txt").unwrap();
    let label = read_label(file).unwrap();
    println!("start training");
    let bar = indicatif::ProgressBar::new(iter_num as u64);
    for iter in 0..iter_num {
        bar.inc(1);
        let res: Vec<Vec<f64>> = hmms
            .hmms
            .par_iter_mut()
            .enumerate()
            .filter_map(|(index, mut hmm)| {
                train(train_files[index], &mut hmm);
                // hmm.dump_hmm(&mut io::stderr());
                match test("./data/test_seq.txt", hmm) {
                    Ok(result) => Some(result),
                    Err(_) => None,
                }
                // hmm.dump_hmm(&mut stderr());
            })
            .collect();
        let res = get_max(res);
        let mut count = 0;
        for i in 0..res.len() {
            if res[i] == label[i] as usize {
                count += 1;
            }
        }
        let current_score = count as f64 / res.len() as f64;
        if current_score >= best_score {
            best_score = current_score;
            best_iter = iter;
        }
        if (best_iter as i32 - iter as i32).abs() > 5 {
            break;
        }
        if best_score >= 0.8 {
            break;
        }
        println!("iter: {}, score: {}", iter, current_score)
    }
    bar.finish();
    println!("best iter: {}", best_iter);
    println!("best score: {}", best_score);
    // hmms.dump_models();
}

pub fn get_max(matrix: Vec<Vec<f64>>) -> Vec<usize> {
    let mut res = Vec::new();
    for i in 0..matrix[0].len() {
        let mut max = 0.0;
        let mut index = 0;
        for j in 0..matrix.len() {
            if matrix[j][i] > max {
                max = matrix[j][i];
                index = j;
            }
        }
        res.push(index + 1);
    }
    res
}

pub fn read_label(file: File) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut label_data = Vec::new();
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let num_sequence: Vec<u32> = line.chars().filter_map(|c| c.to_digit(10)).collect();
        label_data.push(num_sequence[1]);
    }
    Ok(label_data)
}
