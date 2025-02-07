
namespace harp {

class HydrogenCIAImpl : public AttenuatorImpl,
                        public torch::nn::Cloneable<HydrogenCIAImpl> {
  //! extinction x-section + single scattering albedo + phase function moments
  //! (batch, specs, temps, levels, comps)
  torch::Tensor kdata_h2h2;
  torch::Tensor kdata_h2he;

  //! scale the atmospheric variables to the standard grid
  AtmToStandardGrid scale_grid;

  //! Constructor to initialize the layer
  HydrogenCIAImpl();
  explicit HydrogenCIAImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Load opacity from data file
  virtual void load();

  //! Get optical properties
  torch::Tensor forward(torch::Tensor var_x);
};
TORCH_MODULE(HydrogenCIA);

}  // namespace harp
