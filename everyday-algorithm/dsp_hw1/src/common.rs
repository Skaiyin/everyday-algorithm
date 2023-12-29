use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead},
};

pub struct Data {
    pub num_sequence: Vec<Vec<u32>>,
}

pub fn read_data(path: &str) -> Result<Data, Box<dyn std::error::Error>> {
    //从文件读取
    let mut data = Data {
        num_sequence: Vec::new(),
    };
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);

    let char_to_num = [('A', 0), ('B', 1), ('C', 2), ('D', 3), ('E', 4), ('F', 5)]
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();

    for line in reader.lines() {
        let line = line?;
        let num_sequence: Vec<_> = line
            .chars()
            .filter_map(|c| char_to_num.get(&c))
            .cloned()
            .collect();
        data.num_sequence.push(num_sequence);
    }
    Ok(data)
}
