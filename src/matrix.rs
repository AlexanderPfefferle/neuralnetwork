use crate::xorshift::Xorshift32;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Box<[f32]>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![0.0; rows * cols].into_boxed_slice(),
        }
    }

    pub fn new_random(rows: usize, cols: usize, rng: &mut Xorshift32) -> Matrix {
        let mut m = Matrix::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                m[r][c] = rng.float() * 2.0 - 1.0;
            }
        }
        m
    }

    pub fn mul_components(first: &Matrix, second: &Matrix) -> Matrix {
        if first.rows != second.rows || first.cols != second.cols {
            panic!(
                "Error: Cant multiply components of matrices with sizes {}x{} and {}x{}!",
                first.rows, first.cols, second.rows, second.cols
            );
        }
        let mut m = Matrix::new(first.rows, first.cols);
        for r in 0..m.rows {
            for c in 0..m.cols {
                m[r][c] = first[r][c] * second[r][c];
            }
        }
        m
    }

    pub fn transpose(&self) -> Matrix {
        let mut m = Matrix::new(self.cols, self.rows);
        for r in 0..m.rows {
            for c in 0..m.cols {
                m[r][c] = self[c][r];
            }
        }
        m
    }

    pub fn apply_function(&self, f: fn(f32) -> f32) -> Matrix {
        let mut m = Matrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                m[r][c] = f(self[r][c]);
            }
        }
        m
    }

    pub fn index_of_max(&self) -> usize {
        let mut max_value: f32 = f32::NEG_INFINITY;
        let mut max_index: usize = 0;
        for i in 0..self.rows * self.cols {
            if self.data[i] > max_value {
                max_value = self.data[i];
                max_index = i;
            }
        }
        max_index
    }
}

impl std::ops::Index<usize> for Matrix {
    type Output = [f32];
    fn index(&self, row: usize) -> &Self::Output {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

impl std::ops::IndexMut<usize> for Matrix {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        &mut self.data[row * self.cols..(row + 1) * self.cols]
    }
}

impl std::ops::Mul<f32> for Matrix {
    type Output = Matrix;
    fn mul(self, other: f32) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[r][c] += self[r][c] * other;
            }
        }
        result
    }
}

impl std::ops::Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!(
                "Error: Cant multiply matrices with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = Matrix::new(self.rows, other.cols);
        for r in 0..self.rows {
            for c in 0..other.cols {
                for i in 0..self.cols {
                    result[r][c] += self[r][i] * other[i][c];
                }
            }
        }
        result
    }
}

impl std::ops::Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Error: Cant add matrices with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[r][c] = self[r][c] + other[r][c];
            }
        }
        result
    }
}

impl std::ops::AddAssign<Matrix> for Matrix {
    fn add_assign(&mut self, other: Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Error: Cant add matrices with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                self[r][c] += other[r][c];
            }
        }
    }
}

impl std::ops::Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Error: Cant add matrices with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[r][c] = self[r][c] - other[r][c];
            }
        }
        result
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = String::new();
        for r in 0..self.rows {
            s.push_str("[");
            for c in 0..self.cols {
                s.push_str(&self[r][c].to_string());
                s.push_str(" ");
            }
            s.pop();
            s.push_str("]\n");
        }
        write!(f, "{}", s)
    }
}
