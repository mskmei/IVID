import argparse


def main(args):
    from utils.config import load_config, update_config
    from utils.en_train import run_ivid

    config = load_config(args.config)

    if args.recon is not None:
        args.pure_weight = args.recon
        args.bias_weight = args.recon
        args.complete_weight = args.recon

    config = update_config(
        config,
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        text_context_len=args.text_context_len,
        audio_context_len=args.audio_context_len,
        pure_weight=args.pure_weight,
        bias_weight=args.bias_weight,
        complete_weight=args.complete_weight,
        log=args.log,
        save_model=False if args.no_save else None,
    )
    run_ivid(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IVID for multimodal sentiment analysis.")
    parser.add_argument("--config", type=str, default="configs/ivid.yaml")
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"])
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--epochs", type=int, help="maximum training epochs")
    parser.add_argument("--patience", type=int, help="early stopping patience")
    parser.add_argument("--text_context_len", type=int)
    parser.add_argument("--audio_context_len", type=int)
    parser.add_argument(
        "--recon",
        type=float,
        help="deprecated; sets pure, bias, and complete weights together",
    )
    parser.add_argument("--pure_weight", type=float, help="weight for L_pure")
    parser.add_argument("--bias_weight", type=float, help="weight for L_bias")
    parser.add_argument("--complete_weight", type=float, help="weight for L_complete")
    parser.add_argument("--log", type=str)
    parser.add_argument("--no_save", action="store_true", help="do not write best checkpoint")
    main(parser.parse_args())
