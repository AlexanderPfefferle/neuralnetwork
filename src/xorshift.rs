pub struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    pub fn new(seed: u32) -> Xorshift32 {
        Xorshift32 { state: seed }
    }

    fn xorshift32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }

    pub fn float(&mut self) -> f32 {
        self.xorshift32() as f32 / 4_294_967_295u32 as f32
    }
}
