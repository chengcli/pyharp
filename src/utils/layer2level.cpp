
namespace harp {
struct Layer2LevelOptions {
  ADD_ARG(int, k4thOrder) = 0;
  ADD_ARG(bool, use_log) = false;
}

torch::Tensor
layer2level(torch::Tensor var, Layer2LevelOptions const &options) {
  tem_ = hydro_w[0];

  // set temperature at cell interface
  int il = NGHOST, iu = ac.size() - 1 - NGHOST;
  temf_[il] = (3. * tem_[il] - tem_[il + 1]) / 2.;
  temf_[il + 1] = (tem_[il] + tem_[il + 1]) / 2.;
  for (int i = il + 2; i <= iu - 1; ++i)
    temf_[i] = interp_cp4(tem_[i - 2], tem_[i - 1], tem_[i], tem_[i + 1]);
  temf_[iu] = (tem_[iu] + tem_[iu - 1]) / 2.;
  // temf_[iu + 1] = (3. * tem_[iu] - tem_[iu - 1]) / 2.;
  temf_[iu + 1] = tem_[iu];  // isothermal top boundary

  for (int i = 0; i < il; ++i) temf_[i] = tem_[il];
  for (int i = iu + 2; i < ac.size(); ++i) temf_[i] = tem_[iu + 1];

  bool error = false;
  for (int i = 0; i < ac.size(); ++i) {
    if (temf_[i] < 0.) {
      temf_[i] = tem_[i];
      // error = true;
    }
  }
  for (int i = il; i <= iu; ++i) {
    if (tem_[i] < 0.) error = true;
  }
  if (error) {
    for (int i = il; i <= iu; ++i) {
      std::cout << "--- temf[" << i << "] = " << temf_[i] << std::endl;
      std::cout << "tem[" << i << "] = " << tem_[i] << std::endl;
    }
    std::cout << "--- temf[" << iu + 1 << "] = " << temf_[iu + 1] << std::endl;
    throw std::runtime_error("Negative temperature at cell interface");
  }
}
}  // namespace harp
