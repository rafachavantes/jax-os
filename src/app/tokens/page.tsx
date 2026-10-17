import { HermesSection } from "@/components/tokens/HermesSection";
import { SubscriptionCards } from "@/components/tokens/SubscriptionCards";
import { UsageSection } from "@/components/tokens/UsageSection";

export default function TokensPage() {
  return (
    <div className="flex max-w-[1320px] flex-col gap-5">
      <SubscriptionCards />
      <UsageSection />
      <HermesSection />
    </div>
  );
}
