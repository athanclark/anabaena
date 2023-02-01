{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = [ pkgs.gcc pkgs.pkg-config ];
  }
