// One brand presentation = two <img> (dark and light asset). CSS in globals.css shows the one
// matching the server-rendered data-theme and display:none hides the other from sight and from
// assistive tech. Fixed width/height reserve the layout (no shift while the SVG loads).
type BrandImageProps = {
  variant: "logo" | "mark";
  width: number;
  height: number;
  alt?: string; // default is the product name; pass "" where adjacent text already names it
};

export function BrandImage({ variant, width, height, alt = "Jax OS" }: BrandImageProps) {
  return (
    <>
      <img className="brand-dark" src={`/brand/${variant}-dark.svg`} width={width} height={height} alt={alt} />
      <img className="brand-light" src={`/brand/${variant}-light.svg`} width={width} height={height} alt={alt} />
    </>
  );
}
