use ndarray::{Array1};
use serde::{Deserialize, Serialize};

// Defines Leakt version of a Rectified Linear Unit
#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub struct LeakyReLU{
    alpha:f32,
}

impl LeakyReLU{
    // Returns a new LeakyReLU with the given alpha
    #[must_use]
    pub fn new(alpha:f32) -> LeakyReLU{
        LeakyReLU{alpha}
    }

    // Applies the LeakyReLU to the given data
    #[must_use]
    pub fn apply(&self, data:&Array1<f32>) -> Array1<f32>{
        data.mapv(|x| if x >= 0.0 { x } else { self.alpha * x })
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use ndarray::{Array};

    #[test]
    fn test_leakyrelu_simple(){
        let data = Array::from(vec![1.0,2.0,3.0,-1.0,-2.0,-3.0]);
        let expected = Array::from(vec![1.0,2.0,3.0,-0.5,-1.0,-1.5]);
        let leakyrelu = LeakyReLU::new(0.5);
        let result = leakyrelu.apply(&data);
        assert_eq!(result,expected);
    }
}