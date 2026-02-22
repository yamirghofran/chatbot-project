import { ListPlus, Star } from "lucide-react";
import { Link } from "@tanstack/react-router";
import type { ActivityItem } from "@/lib/types";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { TurtleShellIcon } from "@/components/icons/TurtleShellIcon";

export type ActivityFeedProps = {
  items: ActivityItem[];
};

function getInitials(name: string) {
  return name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

function ActivityIcon({ type }: { type: ActivityItem["type"] }) {
  switch (type) {
    case "rating":
      return <Star className="size-3.5 text-[#FFCC00] fill-[#FFCC00] shrink-0" />;
    case "shell_add":
      return <TurtleShellIcon className="size-3.5 text-primary shrink-0" />;
    case "list_add":
      return <ListPlus className="size-3.5 text-muted-foreground shrink-0" />;
  }
}

function ActivityDescription({ item }: { item: ActivityItem }) {
  switch (item.type) {
    case "rating":
      return (
        <>
          rated <span className="font-medium text-foreground">{item.book.title}</span>{" "}
          {item.rating} stars
        </>
      );
    case "shell_add":
      return (
        <>
          added <span className="font-medium text-foreground">{item.book.title}</span> to their shell
        </>
      );
    case "list_add":
      return (
        <>
          added <span className="font-medium text-foreground">{item.book.title}</span> to{" "}
          <span className="font-medium text-foreground">{item.listName}</span>
        </>
      );
  }
}

export function ActivityFeed({ items }: ActivityFeedProps) {
  return (
    <div>
      {items.map((item, i) => (
        <div key={item.id}>
          {i > 0 && <Separator />}
          <div className="flex items-center gap-3 py-3">
            <Avatar size="sm">
              <AvatarFallback>{getInitials(item.user.displayName)}</AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0 text-sm text-muted-foreground">
              <Link
                to="/user/$username"
                params={{ username: item.user.handle }}
                className="font-medium text-foreground hover:underline"
              >
                {item.user.displayName}
              </Link>{" "}
              <ActivityDescription item={item} />
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <ActivityIcon type={item.type} />
              <span className="text-xs text-muted-foreground">{item.timestamp}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
