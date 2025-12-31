import argparse
import numpy as np

from allograph.core.graphio.synth import synthetic_rin
from allograph.core.graphio.pdb import pdb_to_rin
from allograph.core.inference.forward import run_forward
from allograph.core.inference.scan import run_scan
from allograph.core.inference.inverse import run_inverse


def main():
    parser = argparse.ArgumentParser(
        prog="rinet",
        description="RINet command-line interface"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------- FORWARD ----------
    fwd = sub.add_parser("forward", help="Run forward propagation")
    fwd.add_argument("--pdb", type=str, help="PDB file")
    fwd.add_argument("--demo", action="store_true", help="Use synthetic demo graph")
    fwd.add_argument("--seed", type=int, required=True)
    fwd.add_argument("--cutoff", type=float, default=8.0)
    fwd.add_argument("--out", type=str, required=True)

    # ---------- SCAN ----------
    scan = sub.add_parser("scan", help="Run scan over all seed residues")
    scan.add_argument("--pdb", type=str)
    scan.add_argument("--demo", action="store_true")
    scan.add_argument("--cutoff", type=float, default=8.0)
    scan.add_argument("--out", type=str, required=True)

    # ---------- INVERSE ----------
    inv = sub.add_parser("inverse", help="Run inverse inference")
    inv.add_argument("--pdb", type=str)
    inv.add_argument("--demo", action="store_true")
    inv.add_argument("--cutoff", type=float, default=8.0)
    inv.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    if args.demo:
        bundle = synthetic_rin(n=60, seed=0)
    else:
        if not args.pdb:
            parser.error("Must supply --pdb unless using --demo")
        bundle = pdb_to_rin(args.pdb, cutoff=args.cutoff)

    if args.cmd == "forward":
        res = run_forward(bundle, seed_nodes=[args.seed])
        np.savetxt(args.out, res.state, delimiter=",")
    elif args.cmd == "scan":
        res = run_scan(bundle)
        np.savetxt(args.out, res.scores, delimiter=",")
    elif args.cmd == "inverse":
        res = run_inverse(bundle)
        np.savetxt(args.out, res.scores, delimiter=",")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
