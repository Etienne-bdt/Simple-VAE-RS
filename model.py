"""
if __name__ == "__main__":
    LATENT_SIZE = 128
    PATCH_SIZE = 64
    print("Testing model size")
    model = Cond_SRVAE(LATENT_SIZE, PATCH_SIZE)
    x = torch.randn(1, 4, PATCH_SIZE, PATCH_SIZE)
    y = torch.randn(1, 4, PATCH_SIZE // 2, PATCH_SIZE // 2)
    test_x_hat, test_y_hat, test_mu_z, test_logvar_z, test_mu_u, test_logvar_u, test_mu_z_uy, test_logvar_z_uy = model(x, y)

    print(test_x_hat.shape)
    print(test_y_hat.shape)

    assert test_x_hat.shape == x.shape
    assert test_y_hat.shape == y.shape
    print("All size tests passed!")"""
