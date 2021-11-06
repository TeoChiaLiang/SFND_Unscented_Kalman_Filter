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
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  
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

  n_x_ = 5;

  n_aug_ =  7;

  lambda_ = 3 - n_aug_;  

  x_ = Eigen::VectorXd::Zero(n_x_);

  P_ = Eigen::MatrixXd::Identity(n_x_, n_x_);

  Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  Xsig_pred_.fill(0.0);

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

void UKF::AugmentedSigmaPoints(Eigen::MatrixXd& Xsig_aug) {
  // std::cout<<"[AugmentedSigmaPoints] Start"<<std::endl;
  Eigen::VectorXd x_aug = Eigen::VectorXd(n_aug_);
  Eigen::MatrixXd P_aug = Eigen::MatrixXd(n_aug_, n_aug_);
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  Eigen::MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }
  // std::cout<<"[AugmentedSigmaPoints] A: \n"<<A<<std::endl;
  // std::cout<<"[AugmentedSigmaPoints] Xsig_aug: \n"<<Xsig_aug<<std::endl;
  // std::cout<<"[AugmentedSigmaPoints] End"<<std::endl;
}

void UKF::SigmaPointPrediction(const Eigen::MatrixXd& Xsig_aug, const double delta_t) {
  // std::cout<<"[SigmaPointPrediction] Start"<<std::endl;
  // predict sigma points
  for (int i = 0; i< 2*n_aug_+1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    // predicted state values
    double px_p, py_p;
    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;
    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  // std::cout<<"[AugmentedSigmaPoints] Xsig_pred_: \n"<<Xsig_pred_<<std::endl;
  // std::cout<<"[SigmaPointPrediction] End"<<std::endl;
}

void UKF::PredictMeanAndCovariance() {
  //std::cout<<"[PredictMeanAndCovariance] Start"<<std::endl;
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    norm(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
  //std::cout<<"[PredictMeanAndCovariance] End"<<std::endl;
}

void UKF::PredictMeasurementLidar(Eigen::VectorXd& z_out, Eigen::MatrixXd& S_out, Eigen::MatrixXd& Zsig) {
  //std::cout<<"[PredictMeasurementLidar] Start"<<std::endl;
  int n_z = 2;
  // create matrix for sigma points in measurement space
  Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);
  // mean predicted measurement
  Eigen::VectorXd z_pred = Eigen::VectorXd(n_z);
  // measurement covariance matrix S
  Eigen::MatrixXd S = Eigen::MatrixXd(n_z,n_z);
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // measurement model
    Zsig(0,i) = Xsig_pred_(0,i);  // x
    Zsig(1,i) = Xsig_pred_(1,i);  // y
  }
  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // add measurement noise covariance matrix
  Eigen::MatrixXd R = Eigen::MatrixXd(n_z,n_z);
  R <<  std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  S = S + R;
  z_out = z_pred;
  S_out = S;
  //std::cout<<"[PredictMeasurementLidar] End"<<std::endl;
}

void UKF::UpdateStateLidar(const Eigen::MatrixXd& Zsig,      //sigma points in measurement space
                          const Eigen::VectorXd& z_pred,    //predicted measurement mean
                          const Eigen::MatrixXd& S,         //predicted measurement covariance
                          const Eigen::VectorXd& z         //incoming measurement
                          ) {
  // std::cout<<"[UpdateStateLidar] Start"<<std::endl;
  // create matrix for cross correlation Tc
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, Zsig.rows());
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  // Kalman gain K;
  Eigen::MatrixXd K = Tc * S.inverse();
  // residual
  Eigen::VectorXd z_diff = z - z_pred;
  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  norm(x_(3));
  P_ = P_ - K*S*K.transpose();
  // std::cout<<"[UpdateStateLidar] x_: "<<x_.transpose()<<std::endl; 
  // std::cout<<"[UpdateStateLidar] End"<<std::endl;
}

void UKF::PredictMeasurementRadar(Eigen::VectorXd& z_out, Eigen::MatrixXd& S_out, Eigen::MatrixXd& Zsig) {
  int n_z = 3;
  // create matrix for sigma points in measurement space
  Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);
  // mean predicted measurement
  Eigen::VectorXd z_pred = Eigen::VectorXd(n_z);
  // measurement covariance matrix S
  Eigen::MatrixXd S = Eigen::MatrixXd(n_z,n_z);
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }
  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // add measurement noise covariance matrix
  Eigen::MatrixXd R = Eigen::MatrixXd(n_z,n_z);
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  S = S + R;
  z_out = z_pred;
  S_out = S;
}

void UKF::UpdateStateRadar(const Eigen::MatrixXd& Zsig,      //sigma points in measurement space
                          const Eigen::VectorXd& z_pred,    //predicted measurement mean
                          const Eigen::MatrixXd& S,         //predicted measurement covariance
                          const Eigen::VectorXd& z         //incoming measurement
                          ) {
  // create matrix for cross correlation Tc
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, Zsig.rows());
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    // state difference
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  // Kalman gain K;
  Eigen::MatrixXd K = Tc * S.inverse();
  // residual
  Eigen::VectorXd z_diff = z - z_pred;
  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  // std::cout<<"[UpdateStateRadar] x_: "<<x_.transpose()<<std::endl; 
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  //std::cout<<"[ProcessMeasurement] Start **********************************************"<<std::endl;
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    UpdateLidar(meas_package);
  }else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
    UpdateRadar(meas_package);
  }
  //std::cout<<"[ProcessMeasurement] End **********************************************"<<std::endl;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // std::cout<<"[Prediction] Start | "<<delta_t<<" ======================================================"<<std::endl;
  //std::cout<<"[Prediction] [Bef] x_: "<<x_.transpose()<<std::endl;
  // std::cout<<"[Prediction] P_: \n"<<P_<<std::endl;
  if(is_initialized_){
    Eigen::MatrixXd Xsig_aug;
    AugmentedSigmaPoints(Xsig_aug);
    SigmaPointPrediction(Xsig_aug, delta_t);
    PredictMeanAndCovariance(); 
  }
  norm(x_(3));
  // std::cout<<"[Prediction] [Aft] x_: "<<x_.transpose()<<std::endl;
  // std::cout<<"[Prediction] End ======================================================"<<std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  std::cout<<"[UpdateLidar] Start | Z: "<<meas_package.raw_measurements_.transpose()<<std::endl;
  if(is_initialized_){
    std::cout<<"[UpdateLidar] [Bef] x_: "<<x_.transpose()<<std::endl; 
    // std::cout<<"[UpdateLidar] Xsig_pred_: \n"<<Xsig_pred_<<std::endl;
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    Eigen::MatrixXd ZSig;
    PredictMeasurementLidar(z_pred, S, ZSig);
    // std::cout<<"[UpdateLidar] z_pred: "<<z_pred.transpose()<<std::endl;
    // std::cout<<"[UpdateLidar] S: \n"<<S<<std::endl;
    // std::cout<<"[UpdateLidar] ZSig: \n"<<ZSig<<std::endl;
    UpdateStateLidar(ZSig, z_pred, S, meas_package.raw_measurements_);
    std::cout<<"[UpdateLidar] [Aft] x_: "<<x_.transpose()<<std::endl; 
  }else{
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
    is_initialized_ = true;
    std::cout<<"[UpdateLidar] [Init] x_: "<<x_.transpose()<<std::endl; 
  }
  std::cout<<"[UpdateLidar] End"<<std::endl;

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  std::cout<<"[UpdateRadar] Start | Z: "<<meas_package.raw_measurements_.transpose()<<std::endl;
  if(is_initialized_){
    std::cout<<"[UpdateRadar] [Bef] x_: "<<x_.transpose()<<std::endl;
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    Eigen::MatrixXd ZSig;
    PredictMeasurementRadar(z_pred, S, ZSig);
    UpdateStateRadar(ZSig, z_pred, S, meas_package.raw_measurements_);
    std::cout<<"[UpdateRadar] [Aft] x_: "<<x_.transpose()<<std::endl; 
  }else{
    x_(0) = meas_package.raw_measurements_(0)*cos(meas_package.raw_measurements_(1));
    x_(1) = meas_package.raw_measurements_(0)*sin(meas_package.raw_measurements_(1));
    is_initialized_ = true;
    std::cout<<"[UpdateRadar] x_: "<<x_.transpose()<<std::endl; 
  }
  std::cout<<"[UpdateRadar] End"<<std::endl;
}