{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = [
      pkgs.gcc
      pkgs.pkg-config
      pkgs.xorg.libX11
      pkgs.xorg.libX11.dev
      pkgs.xorg.libXcursor
      pkgs.xorg.libXcursor.dev
      pkgs.xorg.libXrandr
      pkgs.xorg.libXrandr.dev
      pkgs.xorg.libXi
      pkgs.xorg.libXi.dev
      pkgs.libGL
      pkgs.libGL.dev
      pkgs.libGLU
      pkgs.libGLU.dev
      pkgs.glew-egl
      pkgs.glew-egl.dev
      pkgs.libglvnd
      pkgs.libglvnd.dev
    ];
  }
