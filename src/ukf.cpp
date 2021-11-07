#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;

  time_us_ = 0;

  n_x_ = 5;

  n_aug_ =  7;

  lambda_ = 3 - n_aug_;  

  x_ = Eigen::VectorXd::Zero(n_x_);

  P_ = Eigen::MatrixXd::Identity(n_x_, n_x_);

  Xsig_pred_ = Eigen::MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
  
  NIS_lidar = 0;

  NIS_radar = 0;

  setWeight();
}

UKF::~UKF() {}

void UKF::setWeight(){
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; ++i) {  
    weights_(i) = weight;
  }
}

void UKF::GenerateSigmaPoints(Eigen::MatrixXd* Xsig_out) {
  Eigen::MatrixXd Xsig = Eigen::MatrixXd(n_x_, 2 * n_x_ + 1);
  Eigen::MatrixXd A = P_.llt().matrixL();
  Xsig.col(0) = x_;
  for (int i = 0; i < n_x_; ++i) {
    Xsig.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
  }
  *Xsig_out = Xsig;
}

void UKF::norm(double& val){
  while (val> M_PI) {
    val-=2.*M_PI;
  }
  while (val<-M_PI) {
    val+=2.*M_PI;
  }
}

void UKF::AugmentSigmaPoints(Eigen::MatrixXd& Xsig_aug) {
  Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(n_aug_);
  x_aug.head(5) = x_;
  // x_aug(5) = 0;
  // x_aug(6) = 0;
  Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  Eigen::MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }
}

void UKF::PredictSigmaPoint(const Eigen::MatrixXd& Xsig_aug, const double delta_t) {
  for (int i = 0; i< 2*n_aug_+1; ++i) {
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    double px_p, py_p;
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    norm(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::PredictMeasurementLidar(Eigen::VectorXd& z_out, Eigen::MatrixXd& S_out, Eigen::MatrixXd& Zsig) {
  int n_z = 2;
  Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }
  Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z,n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i){
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  Eigen::MatrixXd R = Eigen::MatrixXd(n_z,n_z);
  R <<  std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  S = S + R;
  z_out = z_pred;
  S_out = S;
}

void UKF::UpdateStateLidar(const Eigen::MatrixXd& Zsig,      //sigma points in measurement space
                          const Eigen::VectorXd& z_pred,    //predicted measurement mean
                          const Eigen::MatrixXd& S,         //predicted measurement covariance
                          const Eigen::VectorXd& z         //incoming measurement
                          ) {
  Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x_, Zsig.rows());
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
    norm(x_diff(3));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  Eigen::MatrixXd S_Inverse = S.inverse();
  Eigen::MatrixXd K = Tc * S_Inverse;
  Eigen::VectorXd z_diff = z - z_pred;
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_lidar = z_diff.transpose() * S_Inverse * z_diff;
}

void UKF::PredictMeasurementRadar(Eigen::VectorXd& z_out, Eigen::MatrixXd& S_out, Eigen::MatrixXd& Zsig) {
  int n_z = 3;
  Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double p_x    = Xsig_pred_(0,i);
    double p_y    = Xsig_pred_(1,i);
    double v      = Xsig_pred_(2,i);
    double yaw    = Xsig_pred_(3,i);
    double v1     = cos(yaw)*v;
    double v2     = sin(yaw)*v;
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       
    Zsig(1,i) = atan2(p_y,p_x);                                
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);
  }
  Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z,n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    norm(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  Eigen::MatrixXd R = Eigen::MatrixXd(n_z,n_z);
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  S = S + R;
  z_out = z_pred;
  S_out = S;
}

void UKF::UpdateStateRadar(const Eigen::MatrixXd& Zsig,    //sigma points in measurement space
                          const Eigen::VectorXd& z_pred,   //predicted measurement mean
                          const Eigen::MatrixXd& S,        //predicted measurement covariance
                          const Eigen::VectorXd& z         //incoming measurement
                          ) {
  Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x_, Zsig.rows());
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    norm(z_diff(1));
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
    norm(x_diff(3));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  Eigen::MatrixXd S_Inverse = S.inverse();
  Eigen::MatrixXd K = Tc * S_Inverse;
  Eigen::VectorXd z_diff = z - z_pred;
  norm(z_diff(1));
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_radar = z_diff.transpose() * S_Inverse * z_diff;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
	double dt = (meas_package.timestamp_ - time_us_) / 1.0e6;
	time_us_ = meas_package.timestamp_;
	Prediction(dt);
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    UpdateLidar(meas_package);
  }else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  if(is_initialized_){
    Eigen::MatrixXd Xsig_aug;
    AugmentSigmaPoints(Xsig_aug);
    PredictSigmaPoint(Xsig_aug, delta_t);
    PredictMeanAndCovariance(); 
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  if(is_initialized_){
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    Eigen::MatrixXd ZSig;
    PredictMeasurementLidar(z_pred, S, ZSig);
    UpdateStateLidar(ZSig, z_pred, S, meas_package.raw_measurements_);
  }else{
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
    is_initialized_ = true;
  }
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  if(is_initialized_){;
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    Eigen::MatrixXd ZSig;
    PredictMeasurementRadar(z_pred, S, ZSig);
    UpdateStateRadar(ZSig, z_pred, S, meas_package.raw_measurements_);
  }else{
    x_(0) = meas_package.raw_measurements_(0)*cos(meas_package.raw_measurements_(1));
    x_(1) = meas_package.raw_measurements_(0)*sin(meas_package.raw_measurements_(1));
    is_initialized_ = true; 
  }
}