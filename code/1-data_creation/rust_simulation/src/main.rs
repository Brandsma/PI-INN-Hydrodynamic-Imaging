fn wavelet_e(p) -> f64 {
    (1 - 2 * p**2) / ((1 + p**2)**(5 / 2))
}

fn wavelet_o(p) -> f64 {
    (-3 * p) / ((1 + p**2)**(5 / 2))
}

fn wavelet_n(p) -> f64 {
    (2 - p**2) / ((1 + p**2)**(5 / 2))
}

fn v_x(s, x, y, theta, a, norm_w) -> f64 {
    let p = (s - x) / y;
    let C = (norm_w * a**3) / (2 * y**3);
    C * (wavelet_o(p) * math.sin(theta) - wavelet_e(p) * math.cos(theta))
}

fn v_x(s, x, y, theta, a, norm_w) -> f64 {
    let p = (s - x) / y;
    let C = (norm_w * a**3) / (2 * y**3);
    C * (wavelet_n(p) * math.sin(theta) - wavelet_o(p) * math.cos(theta))
}

fn simulate(theta: f32, a: f32, norm_w: f32, lower_bound_sensor: u8, upper_bound_sensor: u8, lower_x: u8, upper_x: u8, lower_y: u8, upper_y: u8) {

}

fn main() {
    println!("Hello, world!");
}
