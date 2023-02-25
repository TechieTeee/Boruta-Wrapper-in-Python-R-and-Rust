extern crate rustlearn;
extern crate serde;
extern crate rayon;

use rustlearn::ensemble::random_forest::{Hyperparameters, RandomForestRegressor};
use rustlearn::feature_extraction::BorutaFeatureSelector;
use rustlearn::prelude::*;
use serde::Deserialize;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::collections::VecDeque;

#[derive(Debug, Deserialize)]
struct Record {
    data: Vec<f32>,
    target: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Load the dataset
    let file = File::open("sales.csv")?;
    let reader = BufReader::new(file);

    let records: VecDeque<Record> = serde_csv::from_reader(reader)?;

    let (data, target): (Vec<Vec<f32>>, Vec<f32>) = records.into_iter()
        .map(|r| (r.data, r.target))
        .unzip();

    // Convert the data to a sparse matrix
    let data = SparseRowArray::from(data);

    // Create a random forest regressor
    let hyperparameters = Hyperparameters::new()
        .max_depth(10)
        .min_samples_split(10);

    let mut rf = RandomForestRegressor::new(hyperparameters);

    // Train the random forest regressor
    rf.fit(&data, &target)?;

    // Create the Boruta feature selector
    let mut boruta_selector = BorutaFeatureSelector::new(&data, &target, None, None)?;

    // Run the Boruta algorithm in parallel
    let num_threads = num_cpus::get();
    let rf = &mut rf;
    boruta_selector.run_in_parallel(num_threads, |data, target| {
        let mut rf = rf.clone();
        rf.fit(&data, &target).unwrap();
        rf
    })?;

    // Print the selected features
    let selected_features = boruta_selector.selected_features();
    let mut writer = csv::Writer::from_writer(std::io::stdout());

    selected_features.into_par_iter().for_each(|feature| {
        writer.write_record(&[&feature.to_string()]).unwrap();
    });

    Ok(())
}
