import { Spinner } from "./Spinner";

export function CenteredLoading() {
  return (
    <div className="grid min-h-[50vh] place-items-center text-foreground">
      <Spinner />
    </div>
  );
}
