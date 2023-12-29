use std::{
    fs::File,
    io::{self, BufRead},
};

pub struct HMM {
    pub model_name: String,
    pub state_num: usize,
    pub observ_num: usize,
    pub initial: Vec<f64>,
    pub transition: Vec<Vec<f64>>,
    pub observation: Vec<Vec<f64>>,
}
pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}
impl HMM {
    pub fn new(model_name: String, path: &str) -> Result<HMM, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);

        let mut initial = Vec::new();
        let mut transition = Vec::new();
        let mut observation = Vec::new();
        let mut current_section = "";
        let mut state_num = 0;
        let mut observ_num = 0;
        for line in reader.lines() {
            let line = line?;
            if line.starts_with("initial:") {
                current_section = "initial";
                let parts: Vec<&str> = line.split_whitespace().collect();
                state_num = parts[1].parse::<usize>().unwrap();
            } else if line.starts_with("transition:") {
                current_section = "transition";
            } else if line.starts_with("observation:") {
                current_section = "observation";
                let parts: Vec<&str> = line.split_whitespace().collect();
                observ_num = parts[1].parse::<usize>().unwrap();
            } else if !line.is_empty() {
                match current_section {
                    "initial" => {
                        initial = line
                            .split_whitespace()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect();
                    }
                    "transition" => {
                        let row: Vec<f64> = line
                            .split_whitespace()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect();
                        transition.push(row);
                    }
                    "observation" => {
                        let row: Vec<f64> = line
                            .split_whitespace()
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect();
                        observation.push(row);
                    }
                    _ => {}
                }
            }
        }

        Ok(HMM {
            model_name: model_name,
            state_num: state_num,
            observ_num: observ_num,
            initial: initial,
            transition: transition,
            observation: transpose(&observation),
        })
    }

    pub fn dump_hmm(&self, stream: &mut dyn io::Write) {
        writeln!(stream, "model name: {}", self.model_name);
        writeln!(stream, "initial: {}", self.state_num);
        writeln!(stream, "transition: {}", self.observ_num);
        writeln!(stream, "observation: {}", self.observ_num);
        writeln!(stream, "initial: {}", self.initial.len());
        for i in 0..self.initial.len() {
            write!(stream, "{} ", self.initial[i]);
        }
        writeln!(stream, "");
        writeln!(
            stream,
            "transition: {} {}",
            self.transition.len(),
            self.transition[0].len()
        );
        for i in 0..self.transition.len() {
            for j in 0..self.transition[i].len() {
                write!(stream, "{} ", self.transition[i][j]);
            }
            writeln!(stream, "");
        }
        writeln!(
            stream,
            "observation: {} {}",
            self.observation.len(),
            self.observation[0].len()
        );
        for i in 0..self.observation.len() {
            for j in 0..self.observation[i].len() {
                write!(stream, "{} ", self.observation[i][j]);
            }
            writeln!(stream, "");
        }
    }
}

pub struct Models {
    pub hmms: Vec<HMM>,
    pub num: usize,
}
impl Models {
    pub fn new(init_file: &str, model_num: usize) -> Result<Models, Box<dyn std::error::Error>> {
        let mut hmms = Vec::new();
        for num in 0..model_num {
            let hmm = HMM::new(num.to_string(), &init_file)?;
            // hmm.dump_hmm(&mut io::stderr());
            hmms.push(hmm);
        }
        let num = hmms.len();
        Ok(Models {
            hmms: hmms,
            num: num,
        })
    }

    pub fn dump_models(&self) {
        for i in 0..self.num {
            self.hmms[i].dump_hmm(&mut io::stderr());
        }
    }
}
